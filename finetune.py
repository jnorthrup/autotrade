#!/usr/bin/env python3
"""
Fine-tune HRM model on a fixed bag for daytrading.

Loads a pretrained checkpoint and fine-tunes on a fixed bag of pairs
from a recent time window. Growth is disabled during fine-tuning.

Example:
    python3 finetune.py --pretrained model_weights_pretrained.pt --bag bag.json --lr 0.0001
"""

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hrm_model import HierarchicalReasoningModel
from coin_graph import CoinGraph, EdgeState, NodeState
from config import Config
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
                _pool_client = False
        except Exception:
            _pool_client = False
    return _pool_client is not False


def _pool() -> PoolClient:
    """Return the shared PoolClient instance (only valid when _use_pool())."""
    return _pool_client


# Default output path for daytrade model
DEFAULT_OUTPUT_PATH = "model_weights_daytrade.pt"
DEFAULT_DEVICE = "cpu"

# Default time window for fine-tuning (last 60 days)
DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_MIN_LOOKBACK_DAYS = 30
DEFAULT_MAX_LOOKBACK_DAYS = 90


def _training_status(total_loss: float, n_updates: int) -> str:
    if n_updates <= 0:
        return "no scored updates; warmup or maturity never completed"
    return f"avg_loss={total_loss / n_updates:.6f}, n_updates={n_updates}"


def _print_profile_summary(profile_stats: Dict[str, float]):
    print("Profile summary:")
    print(
        f"  device={profile_stats.get('device_type', 'unknown')} "
        f"bars={int(profile_stats.get('bars_processed', 0))} "
        f"updates={int(profile_stats.get('n_updates', 0))}"
    )
    print(
        f"  graph.update={profile_stats.get('graph_update_seconds', 0.0):.4f}s "
        f"update_prices={profile_stats.get('update_prices_seconds', 0.0):.4f}s "
        f"predict_prepare={profile_stats.get('predict_prepare_seconds', 0.0):.4f}s "
        f"predict_forward={profile_stats.get('predict_forward_seconds', 0.0):.4f}s"
    )
    print(
        f"  update_prepare={profile_stats.get('update_prepare_seconds', 0.0):.4f}s "
        f"update_forward_backward={profile_stats.get('update_forward_backward_seconds', 0.0):.4f}s "
        f"predict_edges={int(profile_stats.get('predict_edges', 0))} "
        f"update_edges={int(profile_stats.get('update_edges', 0))}"
    )


def _load_bag(bag_path: str) -> List[str]:
    """Load fixed bag of pairs from JSON file."""
    with open(bag_path, 'r') as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs from {bag_path}")
    return pairs


def _get_recent_time_window(
    db_path: str,
    pairs: List[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    granularity: str = "300",
) -> Tuple[datetime, datetime]:
    """Get recent time window for fine-tuning data."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    
    # Verify data availability
    available_pairs = set()
    if _use_pool():
        for pair in pairs:
            try:
                rows = _pool().execute(
                    """SELECT COUNT(*) FROM candles 
                    WHERE product_id = ? AND granularity = ?
                    AND timestamp >= ? AND timestamp < ?""",
                    [pair, granularity, start, end],
                )
                if rows and rows[0][0] > 0:
                    available_pairs.add(pair)
            except Exception:
                pass
    else:
        with duckdb.connect(db_path) as conn:
            for pair in pairs:
                try:
                    row = conn.execute(
                        """SELECT COUNT(*) FROM candles 
                        WHERE product_id = ? AND granularity = ?
                        AND timestamp >= ? AND timestamp < ?""",
                        [pair, granularity, start, end],
                    ).fetchone()
                    if row and row[0] > 0:
                        available_pairs.add(pair)
                except Exception:
                    pass
    
    print(f"Data available for {len(available_pairs)}/{len(pairs)} pairs in {lookback_days}d window")
    return start, end


def _make_subset_graph(
    full_graph: CoinGraph,
    selected_pairs: List[str],
) -> CoinGraph:
    """Create a subgraph with only the selected pairs and their edges."""
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
    else:
        trial.common_timestamps = full_graph.common_timestamps
    return trial


def run_training(
    graph: CoinGraph,
    model: HierarchicalReasoningModel,
    start_bar: int = 0,
    end_bar: Optional[int] = None,
    print_every: int = 100,
    profile_stats: Optional[Dict[str, float]] = None,
) -> Tuple[float, int]:
    """Train model on graph bars. Returns (total_loss, n_updates)."""
    if end_bar is None:
        end_bar = len(graph.common_timestamps)

    model.set_profile_enabled(profile_stats is not None)
    
    ts_to_bar = {ts: i for i, ts in enumerate(graph.common_timestamps)}
    bars_with_data = set()
    for df in graph.edges.values():
        common_ts = set(df.index) & ts_to_bar.keys()
        bars_with_data.update(ts_to_bar[ts] for ts in common_ts)
    
    total_loss = 0.0
    n_updates = 0
    graph_update_seconds = 0.0
    
    sorted_bars = sorted(b for b in bars_with_data if start_bar <= b < end_bar)
    print(f"Training on {len(sorted_bars)} bars with data")
    
    loop_start = time.perf_counter() if profile_stats is not None else None
    for i, bar_idx in enumerate(sorted_bars):
        if bar_idx >= len(graph.common_timestamps):
            break

        graph_update_start = time.perf_counter() if profile_stats is not None else None
        edge_accels, edge_velocities, hit_ptt, hit_stop = graph.update(bar_idx)
        if graph_update_start is not None:
            graph_update_seconds += time.perf_counter() - graph_update_start
        
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
        
        if i % print_every == 0 and i > 0:
            if n_updates > 0:
                avg_loss = total_loss / n_updates
                print(f"Bar {bar_idx}: avg_loss={avg_loss:.6f}, n_updates={n_updates}")
            else:
                print(f"Bar {bar_idx}: warmup, candle history/prediction queues not full yet")

    if profile_stats is not None:
        profile_stats.clear()
        profile_stats.update(model.get_profile_stats())
        profile_stats['graph_update_seconds'] = graph_update_seconds
        profile_stats['bars_processed'] = float(len(sorted_bars))
        profile_stats['bars_with_data'] = float(len(sorted_bars))
        profile_stats['n_updates'] = float(n_updates)
        profile_stats['loop_seconds'] = time.perf_counter() - loop_start if loop_start is not None else 0.0
    
    return total_loss, n_updates


def finetune(
    pretrained_path: str,
    bag_path: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    learning_rate: float = 0.0001,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    exchange: str = 'coinbase',
    skip_fetch: bool = True,
    profile: bool = False,
    device: str = DEFAULT_DEVICE,
) -> None:
    """
    Fine-tune pretrained HRM model on fixed bag.
    
    Args:
        pretrained_path: Path to pretrained checkpoint
        bag_path: Path to fixed bag JSON file
        output_path: Path to save fine-tuned model
        learning_rate: Learning rate for fine-tuning (lower than pretraining)
        lookback_days: Number of recent days to fine-tune on
        exchange: Exchange name ('coinbase' or 'binance')
        skip_fetch: Skip fetching new data (use cached)
    """
    print("=" * 60)
    print("HRM Fine-Tuning for Daytrading")
    print("=" * 60)
    
    # Load fixed bag
    selected_pairs = _load_bag(bag_path)
    if not selected_pairs:
        raise ValueError("No pairs loaded from bag")
    
    print(f"\nUsing fixed bag: {len(selected_pairs)} pairs")
    print(f"Fine-tuning on last {lookback_days} days of data")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Growth: DISABLED (no expansion during fine-tuning)")
    
    # Load coin graph with full data
    print("\nLoading coin graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load(
        lookback_days=lookback_days + 30,  # Extra buffer for history
        min_partners=5,
        max_partners=None,
        exchange=exchange,
        skip_fetch=skip_fetch,
    )
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges, {n_bars} bars")
    
    if n_bars == 0:
        print("No data available. Run fetch_candles.py first.")
        return
    
    # Create subset graph with only bag pairs
    print("\nCreating subgraph from bag...")
    trial_graph = _make_subset_graph(graph, selected_pairs)
    if not trial_graph.edges:
        raise ValueError("No edges available in bag subgraph")
    print(f"Subgraph: {len(trial_graph.edges)} edges")
    
    # Initialize model from pretrained checkpoint
    # Note: h_dim, H_layers, L_layers are loaded from checkpoint
    print(f"\nLoading pretrained checkpoint: {pretrained_path}")
    
    # Create model with defaults - will be overwritten by checkpoint
    model = HierarchicalReasoningModel(
        n_edges=len(trial_graph.edges),
        learning_rate=learning_rate,  # Fine-tuning uses lower LR
        y_depth=200,
        x_pixels=20,
        curvature=2.0,
        h_dim=4,  # Will be overwritten by checkpoint
        z_dim=4,
        prediction_depth=1,
        H_layers=2,  # Will be overwritten by checkpoint
        L_layers=2,  # Will be overwritten by checkpoint
        device=device,
    )
    model.register_edges(list(trial_graph.edges.keys()))
    
    # Load pretrained weights
    if Path(pretrained_path).exists():
        model.load(pretrained_path)
        print(f"Loaded pretrained weights from {pretrained_path}")
        print(f"  h_dim={model.h_dim}, H_layers={model.H_layers}, L_layers={model.L_layers}")
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    
    # Override learning rate for fine-tuning
    model._lr = learning_rate
    if model._optimizer is not None:
        for param_group in model._optimizer.param_groups:
            param_group['lr'] = learning_rate
    print(f"Set fine-tuning learning rate: {learning_rate}")
    
    total_bars = len(trial_graph.common_timestamps)
    split = plan_walk_forward_split(total_bars, model)
    if split is None:
        raise ValueError("Not enough bars for walk-forward fine-tuning validation")
    start_bar, train_end_bar, validation_end_bar = split
    
    print(
        f"\nFine-tuning on bars {start_bar}:{train_end_bar} "
        f"with walk-forward holdout {train_end_bar}:{validation_end_bar}"
    )
    
    # Run fine-tuning (no growth - fixed architecture)
    print("\nStarting fine-tuning...")
    profile_stats = {} if profile else None
    total_loss, n_updates = run_training(
        trial_graph, model,
        start_bar=start_bar,
        end_bar=train_end_bar,
        print_every=100,
        profile_stats=profile_stats,
    )
    walk_forward_loss, walk_forward_updates = run_walk_forward_validation(
        trial_graph,
        model,
        validation_start_bar=train_end_bar,
        end_bar=validation_end_bar,
        warmup_start_bar=start_bar,
    )
    
    print(f"\nFine-tuning complete: {_training_status(total_loss, n_updates)}")
    if walk_forward_loss is not None:
        print(
            f"Walk-forward holdout: avg_loss={walk_forward_loss:.6f}, "
            f"n_updates={walk_forward_updates}"
        )
    else:
        print("Walk-forward holdout: no scored updates")
    if profile_stats is not None:
        _print_profile_summary(profile_stats)
    
    # Save with timestamp metadata
    saved_checkpoint = False
    if n_updates > 0 and walk_forward_loss is not None:
        print(f"\nSaving fine-tuned model to: {output_path}")
        model.save(output_path, checkpoint_type="finetuned_daytrade")
        saved_checkpoint = True
    else:
        print(f"\nSkipping save to {output_path}: walk-forward validation did not score")
    
    print("=" * 60)
    if saved_checkpoint:
        print(f"Saved fine-tuned model to: {output_path}")
    print(f"  pretrained_from: {pretrained_path}")
    print(f"  bag: {bag_path}")
    print("  promotion: disabled for fixed-bag fine-tune path")
    print(f"  learning_rate: {learning_rate}")
    print(f"  lookback_days: {lookback_days}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune HRM model on fixed bag for daytrading"
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        required=True,
        help='Path to pretrained checkpoint (e.g., model_weights_pretrained.pt)'
    )
    parser.add_argument(
        '--bag',
        type=str,
        required=True,
        help='Path to fixed bag JSON file (e.g., bag.json)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate for fine-tuning (default: 0.0001, lower than pre-train)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f'Output path for fine-tuned model (default: {DEFAULT_OUTPUT_PATH})'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f'Days of recent data to fine-tune on (default: {DEFAULT_LOOKBACK_DAYS}, range: {DEFAULT_MIN_LOOKBACK_DAYS}-{DEFAULT_MAX_LOOKBACK_DAYS})'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default='coinbase',
        choices=['coinbase', 'binance'],
        help='Exchange to use for data (default: coinbase)'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        default=True,
        help='Skip fetching new data, use cached (default: True)'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Collect per-stage timing for the training loop'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=DEFAULT_DEVICE,
        choices=['auto', 'cpu', 'mps', 'cuda'],
        help=f"Runtime device (default: {DEFAULT_DEVICE})"
    )
    
    args = parser.parse_args()
    
    # Validate lookback range
    if args.lookback_days < DEFAULT_MIN_LOOKBACK_DAYS or args.lookback_days > DEFAULT_MAX_LOOKBACK_DAYS:
        print(f"Warning: lookback_days should be between {DEFAULT_MIN_LOOKBACK_DAYS} and {DEFAULT_MAX_LOOKBACK_DAYS}")
    
    finetune(
        pretrained_path=args.pretrained,
        bag_path=args.bag,
        output_path=args.output,
        learning_rate=args.lr,
        lookback_days=args.lookback_days,
        exchange=args.exchange,
        skip_fetch=args.skip_fetch,
        profile=args.profile,
        device=args.device,
    )


if __name__ == "__main__":
    main()
