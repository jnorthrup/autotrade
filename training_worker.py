#!/usr/bin/env python3
"""
Training Worker - Runs concurrent training sessions in Docker.

Two modes:
  --mode pretrain: Stochastic bags, long history, continuous training
  --mode finetune: Fixed bag, recent window, periodic retraining

Communicates via:
  - DuckDB: Loads candle data
  - Checkpoint files: Loads/saves model weights
  - Logs: JSON metrics output

Example:
  python3 training_worker.py --mode pretrain --worker-id worker-1
  python3 training_worker.py --mode finetune --lookback-days 60
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from hrm_model import HierarchicalReasoningModel
from coin_graph import CoinGraph, EdgeState, NodeState
from config import Config
from candle_cache import CandleCache
from graph_showdown import plan_walk_forward_split, run_walk_forward_validation

import duckdb

from pool_client import PoolClient, pool_is_running as _pool_is_running

# ---------------------------------------------------------------------------
# Pool-routing helpers
# ---------------------------------------------------------------------------

_pool_client = None


def _use_pool() -> bool:
    """Return True if the DuckDB pool server is available."""
    global _pool_client
    if _pool_client is None:
        try:
            if _pool_is_running():
                _pool_client = PoolClient()
            else:
                _pool_client = False  # sentinel: checked, not available
        except Exception:
            _pool_client = False
    return _pool_client is not False


def _pool() -> PoolClient:
    """Return the shared PoolClient instance (only valid when _use_pool())."""
    return _pool_client


# Configuration from environment
DB_PATH = Path(os.getenv("DB_PATH", "candles.duckdb"))
BAG_PATH = Path(os.getenv("BAG_PATH", "bag.json"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "./checkpoints"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
WORKER_ID = os.getenv("WORKER_ID", "training-worker")
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "1000"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "60"))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")

# Scale bag size with model dimensions
BAG_SIZE_SCALE = {
    4: 5,
    16: 20,
    64: 40,
    256: 80,
}

# Square cube progression (powers of 4)
SQUARE_CUBE_SIZES = [4, 16, 64, 256]
GROWTH_CYCLE = ['h', 'H', 'L']  # hidden_size -> H_layers -> L_layers
PLATEAU_WINDOW = 100
PLATEAU_THRESHOLD = 1e-5
PLATEAU_PATIENCE = 3


def _training_status(avg_loss: Optional[float], n_updates: int) -> str:
    if avg_loss is None or n_updates <= 0:
        return "no scored updates; warmup or maturity never completed"
    return f"avg_loss={avg_loss:.6f}, n_updates={n_updates}"


def _next_square_cube_size(size: int) -> Optional[int]:
    if size not in SQUARE_CUBE_SIZES:
        raise ValueError(f"Invalid square-cube size {size}; expected one of {SQUARE_CUBE_SIZES}")
    idx = SQUARE_CUBE_SIZES.index(size)
    if idx + 1 >= len(SQUARE_CUBE_SIZES):
        return None
    return SQUARE_CUBE_SIZES[idx + 1]


def _validate_square_cube_state(hidden_size: int, H_layers: int, L_layers: int):
    sizes = (hidden_size, H_layers, L_layers)
    invalid = [size for size in sizes if size not in SQUARE_CUBE_SIZES]
    if invalid:
        raise ValueError(f"Square-cube state must use only {SQUARE_CUBE_SIZES}, got {sizes}")
    if len(set(sizes)) > 2:
        raise ValueError(f"Square-cube state may use at most 2 distinct powers of 4, got {sizes}")


def _apply_growth_step(
    model: HierarchicalReasoningModel,
    growth_dim: str,
    hidden_size: int,
    H_layers: int,
    L_layers: int,
) -> Tuple[int, int, int, bool]:
    _validate_square_cube_state(hidden_size, H_layers, L_layers)

    if growth_dim == 'h':
        if _next_square_cube_size(hidden_size) is None:
            return hidden_size, H_layers, L_layers, False
        model.grow('h')
        hidden_size = model.h_dim
    elif growth_dim == 'H':
        if _next_square_cube_size(H_layers) is None:
            return hidden_size, H_layers, L_layers, False
        model.grow('H')
        H_layers = model.H_layers
    elif growth_dim == 'L':
        if _next_square_cube_size(L_layers) is None:
            return hidden_size, H_layers, L_layers, False
        model.grow('L')
        L_layers = model.L_layers
    else:
        raise ValueError(f"Unknown growth dimension: {growth_dim}")

    _validate_square_cube_state(hidden_size, H_layers, L_layers)
    return hidden_size, H_layers, L_layers, True


def _is_converged(losses: List[float]) -> bool:
    n = len(losses)
    if n < PLATEAU_WINDOW * PLATEAU_PATIENCE:
        return False
    for i in range(PLATEAU_PATIENCE):
        chunk = losses[-PLATEAU_WINDOW * (PLATEAU_PATIENCE - i):]
        if len(chunk) < PLATEAU_WINDOW:
            continue
        recent = np.mean(chunk[-PLATEAU_WINDOW:])
        older = np.mean(chunk[:PLATEAU_WINDOW])
        if abs(recent - older) > PLATEAU_THRESHOLD:
            return False
    return True


class TrainingWorker:
    """Continuous training worker with checkpoint management."""
    
    def __init__(
        self,
        mode: str,  # 'pretrain' or 'finetune'
        worker_id: str = WORKER_ID,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        save_every: int = SAVE_EVERY,
        poll_interval: int = POLL_INTERVAL,
        lookback_days: int = LOOKBACK_DAYS,
        device: Optional[str] = None,
    ):
        self.mode = mode
        self.worker_id = worker_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every = save_every
        self.poll_interval = poll_interval
        self.lookback_days = lookback_days
        self.device = device or ("cpu" if mode == "finetune" else MODEL_DEVICE)
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # State
        self.model = None
        self.current_graph = None
        self.phase = 0
        self.stopped = False
        
        # Growth state (pretrain only)
        self.growth_idx = 0
        self.hidden_size = SQUARE_CUBE_SIZES[0]
        self.H_layers = SQUARE_CUBE_SIZES[0]
        self.L_layers = SQUARE_CUBE_SIZES[0]
        
        # Fixed bag (finetune only)
        self.fixed_bag = None
        
        # Signal handling
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _setup_logging(self):
        """Setup structured JSON logging."""
        log_file = LOG_DIR / f"{self.worker_id}.{self.mode}.log"
        
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z'
        )
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        
        self.logger = logging.getLogger(self.worker_id)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)
        
        self.logger.info(
            f"TrainingWorker initialized: mode={self.mode}, id={self.worker_id}, device={self.device}"
        )
    
    def _handle_signal(self, signum, frame):
        """Graceful shutdown on signal."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stopped = True
    
    def _get_checkpoint_path(self, checkpoint_type: str = None) -> Path:
        """Get checkpoint path based on mode and type."""
        if self.mode == "pretrain":
            name = "model_weights_pretrained.pt"
        elif checkpoint_type == "pretrained":
            name = "model_weights_pretrained.pt"
        else:
            name = f"model_weights_{self.worker_id}.pt"
        return self.checkpoint_dir / name
    
    def _load_bag_pairs(self) -> List[str]:
        """Load fixed bag for finetuning."""
        with open(BAG_PATH, 'r') as f:
            pairs = json.load(f)
        self.logger.info(f"Loaded {len(pairs)} pairs from {BAG_PATH}")
        return pairs
    
    def _list_available_pairs(self) -> List[str]:
        """List all pairs available in DB."""
        try:
            if _use_pool():
                rows = _pool().execute(
                    "SELECT DISTINCT product_id FROM candles"
                )
            else:
                with duckdb.connect(str(DB_PATH)) as conn:
                    rows = conn.execute("""
                        SELECT DISTINCT product_id FROM candles
                    """).fetchall()
            pairs = [r[0] for r in rows]
            self.logger.info(f"Found {len(pairs)} pairs in database")
            return pairs
        except Exception as e:
            self.logger.warning(f"Could not list pairs: {e}")
            return []
    
    def _compute_volatility_filter(
        self,
        db_path: str,
        pairs: List[str],
        window: int = 100,
        min_velocity: float = 0.001,
    ) -> List[str]:
        """Filter pairs by volatility - keep only high-volatility crypto pairs."""
        filtered = []
        
        for pair in pairs:
            if "USD-" in pair or "USD" in pair or "EUR" in pair or "GBP" in pair or "JPY" in pair:
                # Skip fiat-fiat pairs
                if pair.count("USD") + pair.count("EUR") + pair.count("GBP") >= 2:
                    continue
            
            try:
                if _use_pool():
                    rows = _pool().execute(
                        f"""SELECT close FROM candles
                        WHERE product_id = ?
                        ORDER BY timestamp DESC
                        LIMIT {window + 1}""",
                        [pair],
                    )
                else:
                    with duckdb.connect(str(db_path)) as conn:
                        rows = conn.execute(f"""
                            SELECT close FROM candles
                            WHERE product_id = ?
                            ORDER BY timestamp DESC
                            LIMIT {window + 1}
                        """, [pair]).fetchall()
                
                if len(rows) < window:
                    continue
                    
                closes = np.array([r[0] for r in rows])
                returns = np.abs(np.diff(np.log(closes)))
                mean_velocity = np.mean(returns)
                
                if mean_velocity >= min_velocity:
                    filtered.append(pair)
                    
            except Exception:
                pass
                
        return filtered
    
    def _sample_stochastic_bag(
        self,
        all_pairs: List[str],
        hidden_size: int,
        rng: random.Random,
    ) -> List[str]:
        """Sample bag size scaled by model dimension."""
        bag_size = BAG_SIZE_SCALE.get(hidden_size, 20)
        bag_size = min(bag_size, len(all_pairs))
        bag_size = max(5, bag_size)
        return rng.sample(all_pairs, bag_size)
    
    def _make_subset_graph(
        self,
        full_graph: CoinGraph,
        selected_pairs: List[str],
        start_bar: int,
        end_bar: int,
    ) -> CoinGraph:
        """Create subgraph with only selected pairs."""
        trial = CoinGraph(
            fee_rate=full_graph.fee_rate,
            min_pair_coverage=getattr(full_graph, "min_pair_coverage", 0.9),
        )
        trial.all_pairs = selected_pairs
        
        pair_to_edges = {}
        for pid in full_graph.all_pairs:
            parts = pid.split("-", 1)
            if len(parts) == 2:
                pair_to_edges[pid] = (parts[0], parts[1])
        
        for pid in selected_pairs:
            if pid not in pair_to_edges:
                continue
            base, quote = pair_to_edges[pid]
            for edge in [(base, quote), (quote, base)]:
                if edge in full_graph.edges:
                    trial.edges[edge] = full_graph.edges[edge]
                    trial.edge_state[edge] = EdgeState()
                    trial.nodes.add(base)
                    trial.nodes.add(quote)
                    trial.node_state.setdefault(base, NodeState())
                    trial.node_state.setdefault(quote, NodeState())
        
        if trial.edges and all(not callable(getattr(df, "index", None)) for df in trial.edges.values()):
            trial._align_timestamps()
            trial.common_timestamps = trial.common_timestamps[start_bar:end_bar]
        else:
            trial.common_timestamps = full_graph.common_timestamps[start_bar:end_bar]
        return trial
    
    def _build_model(
        self,
        n_edges: int,
        hidden_size: int,
        H_layers: int,
        L_layers: int,
        learning_rate: float = 0.001,
    ) -> HierarchicalReasoningModel:
        """Build HRM model with current dimensions."""
        return HierarchicalReasoningModel(
            n_edges=n_edges,
            learning_rate=learning_rate,
            y_depth=200,
            x_pixels=20,
            curvature=2.0,
            h_dim=hidden_size,
            z_dim=hidden_size,
            prediction_depth=1,
            H_layers=H_layers,
            L_layers=L_layers,
            H_cycles=2,
            L_cycles=2,
            device=self.device,
        )
    
    def _run_training_step(
        self,
        graph: CoinGraph,
        model: HierarchicalReasoningModel,
    ) -> Tuple[Optional[float], int, List[float]]:
        """Run one training iteration. Returns (avg_loss_or_none, n_updates, loss_history)."""
        total_loss = 0.0
        n_updates = 0
        loss_history: List[float] = []
        model.set_profile_enabled(False)
        
        for bar_idx in range(len(graph.common_timestamps)):
            if self.stopped:
                break
                
            edge_accels, edge_velocities, hit_ptt, hit_stop = graph.update(bar_idx)
            
            if not edge_accels:
                continue

            model.update_prices(graph, bar_idx)
            
            if model.ready_for_prediction(bar_idx):
                model.predict(graph, bar_idx)
            
            if model.ready_for_update(bar_idx, edge_accels):
                loss = model.update(graph, edge_accels, bar_idx, hit_ptt=hit_ptt, hit_stop=hit_stop)
                if loss is not None:
                    total_loss += loss
                    n_updates += 1
                    loss_history.append(loss)
        
        avg_loss = (total_loss / n_updates) if n_updates > 0 else None
        return avg_loss, n_updates, loss_history
    
    def _run_pretrain_cycle(self):
        """One pretrain cycle: sample bag, train, maybe grow."""
        self.phase += 1
        
        rng = random.Random()
        
        # Load all available pairs
        all_pairs = self._list_available_pairs()
        if not all_pairs:
            self.logger.warning("No pairs available, skipping cycle")
            time.sleep(self.poll_interval)
            return
        
        # Volatility filter
        filtered_pairs = self._compute_volatility_filter(
            DB_PATH, all_pairs, window=100, min_velocity=0.001
        )
        if not filtered_pairs:
            self.logger.warning("No pairs after volatility filter")
            return
        
        # Sample stochastic bag
        selected_pairs = self._sample_stochastic_bag(
            filtered_pairs, self.hidden_size, rng
        )
        
        # Sample time window
        if _use_pool():
            total_bars = _pool().execute(
                "SELECT COUNT(DISTINCT timestamp) FROM candles"
            )[0][0]
        else:
            with duckdb.connect(str(DB_PATH)) as conn:
                total_bars = conn.execute("""
                    SELECT COUNT(DISTINCT timestamp) FROM candles
                """).fetchone()[0]
        
        window_bars = min(10000, total_bars)
        max_start = max(0, total_bars - window_bars)
        start_bar = rng.randint(0, max_start) if max_start > 0 else 0
        end_bar = start_bar + window_bars
        
        self.logger.info(f"Phase {self.phase}: {len(selected_pairs)} pairs, bars {start_bar}-{end_bar}")
        
        # Load full graph
        graph = CoinGraph(fee_rate=0.001)
        graph.load(
            lookback_days=365,
            min_partners=5,
            max_partners=None,
            exchange='coinbase',
            skip_fetch=True,
        )
        
        # Create trial graph
        trial_graph = self._make_subset_graph(graph, selected_pairs, start_bar, end_bar)
        if not trial_graph.edges:
            self.logger.warning("Empty trial graph, skipping")
            return
        
        # Load or create model
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.model = self._build_model(
                n_edges=len(trial_graph.edges),
                hidden_size=self.hidden_size,
                H_layers=self.H_layers,
                L_layers=self.L_layers,
            )
            self.model.load(str(checkpoint_path))
            self.hidden_size = self.model.h_dim
            self.H_layers = self.model.H_layers
            self.L_layers = self.model.L_layers
        else:
            self.logger.info("Creating new model")
            self.model = self._build_model(
                n_edges=len(trial_graph.edges),
                hidden_size=self.hidden_size,
                H_layers=self.H_layers,
                L_layers=self.L_layers,
            )
        
        self.model.register_edges(list(trial_graph.edges.keys()))

        split = plan_walk_forward_split(len(trial_graph.common_timestamps), self.model)
        if split is None:
            self.logger.warning("Not enough bars for walk-forward pretrain validation")
            return
        train_start_bar, train_end_bar, validation_end_bar = split
        
        # Train
        avg_loss, n_updates, loss_history = self._run_training_step(
            self._make_subset_graph(
                trial_graph,
                selected_pairs,
                train_start_bar,
                train_end_bar,
            ),
            self.model,
        )
        validation_loss, validation_updates = run_walk_forward_validation(
            trial_graph,
            self.model,
            validation_start_bar=train_end_bar,
            end_bar=validation_end_bar,
            warmup_start_bar=train_start_bar,
        )
        
        self.logger.info(f"Phase {self.phase}: {_training_status(avg_loss, n_updates)}")
        if validation_loss is not None:
            self.logger.info(
                "Phase %s walk-forward: avg_loss=%.6f, updates=%s",
                self.phase,
                validation_loss,
                validation_updates,
            )
        else:
            self.logger.info("Phase %s walk-forward: no scored updates", self.phase)

        growth_dim = GROWTH_CYCLE[self.growth_idx]
        if _is_converged(loss_history):
            old_h = self.hidden_size
            old_H = self.H_layers
            old_L = self.L_layers
            self.hidden_size, self.H_layers, self.L_layers, did_grow = _apply_growth_step(
                self.model,
                growth_dim,
                self.hidden_size,
                self.H_layers,
                self.L_layers,
            )
            if did_grow:
                self.growth_idx = (self.growth_idx + 1) % len(GROWTH_CYCLE)
                self.logger.info(
                    "Converged -> growth %s [h=%s, H=%s, L=%s] -> [h=%s, H=%s, L=%s]",
                    growth_dim,
                    old_h,
                    old_H,
                    old_L,
                    self.hidden_size,
                    self.H_layers,
                    self.L_layers,
                )
                save_path = self._get_checkpoint_path()
                self.model.save(str(save_path), checkpoint_type=f"pretrain_growth_phase_{self.phase}")
                self.logger.info(f"Saved grown checkpoint: {save_path}")
            else:
                self.logger.info(
                    "Converged but %s is already at max [h=%s, H=%s, L=%s]",
                    growth_dim,
                    self.hidden_size,
                    self.H_layers,
                    self.L_layers,
                )
        
        # Save checkpoint periodically
        if self.phase % self.save_every == 0 and n_updates > 0 and validation_loss is not None:
            save_path = self._get_checkpoint_path()
            self.model.save(str(save_path), checkpoint_type=f"pretrain_phase_{self.phase}")
            self.logger.info(f"Saved checkpoint: {save_path}")
        elif self.phase % self.save_every == 0:
            self.logger.info("Skipped checkpoint save: no scored walk-forward validation in this cycle")
    
    def _run_finetune_cycle(self):
        """One finetune cycle: train on recent window with fixed bag."""
        self.phase += 1
        
        # Load fixed bag
        if self.fixed_bag is None:
            self.fixed_bag = self._load_bag_pairs()
        
        self.logger.info(f"Fine-tune phase {self.phase}: {len(self.fixed_bag)} pairs, {self.lookback_days} days")
        
        # Load recent graph
        graph = CoinGraph(fee_rate=0.001)
        graph.load(
            lookback_days=self.lookback_days + 30,
            min_partners=5,
            max_partners=None,
            exchange='coinbase',
            skip_fetch=True,
        )
        
        # Create subgraph with bag pairs
        trial_graph = self._make_subset_graph(
            graph, self.fixed_bag, 0, len(graph.common_timestamps)
        )
        if not trial_graph.edges:
            self.logger.warning("Empty trial graph")
            return
        
        # Load pretrained checkpoint
        pretrained_path = self._get_checkpoint_path(checkpoint_type="pretrained")
        
        if self.model is None:
            if pretrained_path.exists():
                self.logger.info(f"Loading pretrained: {pretrained_path}")
                self.model = self._build_model(
                    n_edges=len(trial_graph.edges),
                    hidden_size=4,
                    H_layers=2,
                    L_layers=2,
                )
                self.model.load(str(pretrained_path))
                # Lower LR for fine-tuning
                self.model._lr = 0.0001
            else:
                self.logger.warning(f"No pretrained checkpoint at {pretrained_path}")
                return
        
        self.model.register_edges(list(trial_graph.edges.keys()))
        self.model._lr = 0.0001
        if self.model._optimizer is not None:
            for param_group in self.model._optimizer.param_groups:
                param_group['lr'] = 0.0001

        split = plan_walk_forward_split(len(trial_graph.common_timestamps), self.model)
        if split is None:
            self.logger.warning("Not enough bars for walk-forward fine-tune validation")
            return
        train_start_bar, train_end_bar, validation_end_bar = split
        
        # Train (no growth)
        avg_loss, n_updates, _ = self._run_training_step(
            self._make_subset_graph(
                trial_graph,
                self.fixed_bag,
                train_start_bar,
                train_end_bar,
            ),
            self.model,
        )
        validation_loss, validation_updates = run_walk_forward_validation(
            trial_graph,
            self.model,
            validation_start_bar=train_end_bar,
            end_bar=validation_end_bar,
            warmup_start_bar=train_start_bar,
        )
        
        self.logger.info(f"Fine-tune phase {self.phase}: {_training_status(avg_loss, n_updates)}")
        if validation_loss is not None:
            self.logger.info(
                "Fine-tune phase %s walk-forward: avg_loss=%.6f, updates=%s",
                self.phase,
                validation_loss,
                validation_updates,
            )
        else:
            self.logger.info("Fine-tune phase %s walk-forward: no scored updates", self.phase)
        
        # Save checkpoint
        if self.phase % self.save_every == 0 and n_updates > 0 and validation_loss is not None:
            save_path = self._get_checkpoint_path()
            self.model.save(str(save_path), checkpoint_type=f"finetune_phase_{self.phase}")
            self.logger.info(f"Saved checkpoint: {save_path}")
        elif self.phase % self.save_every == 0:
            self.logger.info("Skipped checkpoint save: no scored walk-forward validation in this cycle")
    
    def run(self):
        """Main worker loop."""
        self.logger.info(f"Starting {self.mode} worker")
        
        while not self.stopped:
            try:
                if self.mode == "pretrain":
                    self._run_pretrain_cycle()
                elif self.mode == "finetune":
                    self._run_finetune_cycle()
                else:
                    self.logger.error(f"Unknown mode: {self.mode}")
                    break
                    
            except Exception as e:
                self.logger.exception(f"Error in {self.mode} cycle: {e}")
                
            if not self.stopped:
                time.sleep(self.poll_interval)
        
        self.logger.info("Worker stopped")


def main():
    parser = argparse.ArgumentParser(description="HRM Training Worker")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pretrain", "finetune"],
        help="Training mode: pretrain (stochastic) or finetune (fixed bag)"
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID for logging"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N phases"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Days of recent data for finetuning"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Seconds between cycles"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Runtime device override"
    )
    
    args = parser.parse_args()
    
    worker = TrainingWorker(
        mode=args.mode,
        worker_id=args.worker_id or WORKER_ID,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINT_DIR,
        save_every=args.save_every or SAVE_EVERY,
        poll_interval=args.poll_interval or POLL_INTERVAL,
        lookback_days=args.lookback_days or LOOKBACK_DAYS,
        device=args.device,
    )
    
    worker.run()


if __name__ == "__main__":
    main()
