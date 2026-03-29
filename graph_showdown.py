#!/usr/bin/env python3
import argparse
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hrm_model import HierarchicalReasoningModel
from coin_graph import CoinGraph
from config import Config

import duckdb

# Square cube progression: hidden_size, always powers of 4
SQUARE_CUBE_SIZES = [4, 16, 64, 256]
PLATEAU_WINDOW = 100
PLATEAU_THRESHOLD = 1e-5
PLATEAU_PATIENCE = 3

# Growth cycle for square cube: which dimension leads next
# Cycle: h (hidden_size leads) -> H (H_layers catches up) -> L (L_layers catches up) -> h (cubed, cycle repeats)
GROWTH_CYCLE = ['h', 'H', 'L']


def _next_square_cube_size(size: int) -> Optional[int]:
    """Return the next allowed 4^k size, or None at the ceiling."""
    if size not in SQUARE_CUBE_SIZES:
        raise ValueError(f"Invalid square-cube size {size}; expected one of {SQUARE_CUBE_SIZES}")
    idx = SQUARE_CUBE_SIZES.index(size)
    if idx + 1 >= len(SQUARE_CUBE_SIZES):
        return None
    return SQUARE_CUBE_SIZES[idx + 1]


def _validate_square_cube_state(hidden_size: int, H_layers: int, L_layers: int):
    """Enforce powers-of-4 sizes with at most two distinct values."""
    sizes = (hidden_size, H_layers, L_layers)
    invalid = [size for size in sizes if size not in SQUARE_CUBE_SIZES]
    if invalid:
        raise ValueError(
            f"Square-cube state must use only {SQUARE_CUBE_SIZES}, got {sizes}"
        )
    if len(set(sizes)) > 2:
        raise ValueError(
            f"Square-cube state may use at most 2 distinct powers of 4, got {sizes}"
        )


def _apply_growth_step(
    model: HierarchicalReasoningModel,
    growth_dim: str,
    hidden_size: int,
    H_layers: int,
    L_layers: int,
) -> Tuple[int, int, int, bool]:
    """Apply exactly one 4× growth step if the scheduled dimension can grow."""
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
    """Sustained plateau detection across PATIENCE windows."""
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


def run_training(graph: CoinGraph, model: HierarchicalReasoningModel, start_bar: int = 0,
                 end_bar: Optional[int] = None, print_every: int = 100,
                 loss_history: Optional[List[float]] = None) -> Tuple[float, int, bool, List[float]]:
    """Train model on graph bars. Returns (total_loss, n_updates, early_stopped, loss_history)."""
    if end_bar is None:
        end_bar = len(graph.common_timestamps)
    
    if loss_history is None:
        loss_history = []
    
    ts_to_bar = {ts: i for i, ts in enumerate(graph.common_timestamps)}
    bars_with_data = set()
    for df in graph.edges.values():
        common_ts = set(df.index) & ts_to_bar.keys()
        bars_with_data.update(ts_to_bar[ts] for ts in common_ts)
    
    total_loss = 0.0
    n_updates = 0
    early_stopped = False
    
    sorted_bars = sorted(b for b in bars_with_data if start_bar <= b < end_bar)
    print(f"Training on {len(sorted_bars)} bars with data")
    
    for i, bar_idx in enumerate(sorted_bars):
        if bar_idx >= len(graph.common_timestamps):
            break
        
        edge_accels, edge_velocities, hit_ptt, hit_stop = graph.update(bar_idx)
        
        if not edge_accels:
            continue
        
        if bar_idx >= model.prediction_depth:
            model.predict(graph, bar_idx)
        
        if bar_idx >= model.prediction_depth * 2:
            loss = model.update(graph, edge_accels, bar_idx, hit_ptt=hit_ptt, hit_stop=hit_stop)
            if loss is not None:
                total_loss += loss
                n_updates += 1
                loss_history.append(loss)
        
        if i % print_every == 0 and i > 0:
            avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
            print(f"Bar {bar_idx}: avg_loss={avg_loss:.6f}")
    
    return total_loss, n_updates, early_stopped, loss_history


def _list_all_binance_pairs(db_path: str) -> List[str]:
    """List all Binance-style pairs available in candle_cache (DuckDB)."""
    try:
        with duckdb.connect(db_path) as conn:
            rows = conn.execute("SELECT DISTINCT product_id FROM candles").fetchall()
            pairs = [r[0] for r in rows]
            # Filter: must look like BASE-QUOTE with two parts
            pairs = [p for p in pairs if "-" in p and len(p.split("-", 1)) == 2]
            return sorted(set(pairs))
    except Exception as e:
        print(f"[_list_all_binance_pairs] error: {e}")
        return []


FIAT_CURRENCIES = {
    "USD", "USDT", "USDC", "EUR", "GBP", "SGD", "JPY", "BRL", "MXN",
    "TRY", "IDR", "PLN", "ARS", "ZAR", "UAH", "COP", "RUB", "NGN", "EURI",
}


def _compute_volatility_filter(
    db_path: str,
    all_pairs: List[str],
    lookback_days: int = 365,
    granularity: str = "300",
    min_velocity: float = 0.001,
) -> List[str]:
    """Filter pairs by mean |velocity| (log close/open) and drop fiat-fiat edges.

    Returns pairs with mean |velocity| >= min_velocity and no fiat-fiat edges.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    filtered = []
    with duckdb.connect(db_path) as conn:
        for pid in all_pairs:
            parts = pid.split("-", 1)
            if len(parts) != 2:
                continue
            base, quote = parts
            # Drop fiat-fiat edges
            if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
                continue
            # Compute mean |log(close/open)| over the lookback window
            try:
                row = conn.execute(
                    """SELECT AVG(ABS(LN(close / NULLIF(open, 0))))
                       FROM candles
                       WHERE product_id = ? AND granularity = ?
                         AND timestamp BETWEEN ? AND ?
                         AND open > 0 AND close > 0""",
                    [pid, granularity, start, end],
                ).fetchone()
                mean_vel = row[0] if row and row[0] is not None else 0.0
            except Exception:
                mean_vel = 0.0

            if mean_vel >= min_velocity:
                filtered.append(pid)

    return sorted(set(filtered))


# Model-size to bag-size scaling: 4->5, 16->20, 64->40, 256->80
_BAG_SIZE_SCALE = {4: 5, 16: 20, 64: 40, 256: 80}


def _stochastic_bag_sample(
    filtered_pairs: List[str],
    model_size: int,
    rng: random.Random,
    min_pairs: int = 5,
    max_pairs: Optional[int] = None,
) -> List[str]:
    """Sample a stochastic bag of pairs, size scaled by model dimensions.

    Bag size = _BAG_SIZE_SCALE.get(model_size, model_size * 3 / 4), clamped to
    [min_pairs, max_pairs or len(filtered_pairs)].
    """
    if not filtered_pairs:
        return []

    target = _BAG_SIZE_SCALE.get(model_size, max(min_pairs, model_size * 3 // 4))
    if max_pairs is not None:
        target = min(target, max_pairs)
    target = max(min_pairs, min(target, len(filtered_pairs)))

    # Build adjacency for connected subgraph sampling
    adj = _build_pair_adjacency(filtered_pairs)
    return _select_related_pairs(filtered_pairs, adj, target, rng)


def _build_pair_adjacency(all_pairs: List[str]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for pid in all_pairs:
        parts = pid.split("-", 1)
        if len(parts) != 2:
            continue
        for currency in parts:
            adj.setdefault(currency, []).append(pid)
    return adj


def _select_related_pairs(all_pairs: List[str], adj: Dict[str, List[str]],
                          n_pairs: int, rng: random.Random) -> List[str]:
    if n_pairs >= len(all_pairs):
        return list(all_pairs)

    currencies = list(adj.keys())
    if not currencies:
        return list(all_pairs)[:n_pairs]

    selected = set()
    seed_currency = rng.choice(currencies)
    frontier = [seed_currency]
    visited_currencies = {seed_currency}

    while len(selected) < n_pairs and frontier:
        curr = frontier.pop(0)
        candidates = [p for p in adj.get(curr, []) if p not in selected]
        rng.shuffle(candidates)
        for pid in candidates:
            if len(selected) >= n_pairs:
                break
            selected.add(pid)
            parts = pid.split("-", 1)
            for c in parts:
                if c not in visited_currencies:
                    visited_currencies.add(c)
                    frontier.append(c)

        if not frontier and len(selected) < n_pairs:
            remaining = [c for c in currencies if c not in visited_currencies]
            if remaining:
                new_seed = rng.choice(remaining)
                frontier.append(new_seed)
                visited_currencies.add(new_seed)

    return list(selected)


def _make_trial_graph(full_graph: CoinGraph, selected_pairs: List[str],
                      start_bar: int, end_bar: int) -> CoinGraph:
    from coin_graph import EdgeState, NodeState
    
    trial = CoinGraph(fee_rate=full_graph.fee_rate)
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

    trial.common_timestamps = full_graph.common_timestamps[start_bar:end_bar]
    return trial


def run_autoresearch(graph: CoinGraph, db_path: str = 'candles.duckdb',
                     exchange: str = 'coinbase', pm_mode: str = 'single_asset'):
    """
    Autoresearch with Square Cube Progression + Stochastic Bag Sampling.

    Starts with tiny HRM (hidden_size=4, H_layers=4, L_layers=4).
    Trains until plateau, then grows one dimension by 4× using rotational expansion.

    Each iteration:
      1. Load all available pairs from candle_cache (or fall back to graph.all_pairs)
      2. Volatility filter: drop low-velocity and fiat-fiat pairs
      3. Stochastic bag: randomly sample N pairs, size scaled by model dimensions
      4. Stochastic time window: randomly sample (start_bar, end_bar)

    Growth cycle: h -> H -> L -> h (hidden_size leads, then layers catch up)
    Square sizes: 4 -> 16 -> 64 -> 256 (always powers of 4)
    """
    print("Using HRMEdgePredictor for hierarchical reasoning")

    conn = duckdb.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            timestamp TIMESTAMP DEFAULT now(),
            val_bpb DOUBLE,
            params VARCHAR,
            bag_spec VARCHAR,
            growth_phase VARCHAR
        )
    """)

    best_bpb = float('inf')
    best_params = None

    total_bars = len(graph.common_timestamps)
    rng = random.Random()

    # --- Stochastic bag: load all available pairs from DB, then volatility-filter ---
    all_db_pairs = _list_all_binance_pairs(db_path)
    if all_db_pairs:
        print(f"[StochasticBag] Found {len(all_db_pairs)} pairs in candle_cache")
        filtered_pairs = _compute_volatility_filter(
            db_path, all_db_pairs,
            lookback_days=365 if exchange == "coinbase" else 1095,
            granularity="300",
            min_velocity=0.001,
        )
        print(f"[StochasticBag] After volatility filter: {len(filtered_pairs)} pairs")
    else:
        # Fallback to graph.all_pairs (from bag.json)
        filtered_pairs = list(graph.all_pairs)
        print(f"[StochasticBag] No DB pairs found, using graph.all_pairs ({len(filtered_pairs)} pairs)")

    if not filtered_pairs:
        print("[StochasticBag] ERROR: no pairs available after filtering")
        return None

    total_pool = len(filtered_pairs)
    MIN_PAIRS = 5
    MAX_PAIRS = total_pool
    MIN_WINDOW_BARS = max(200, total_bars // 20)
    MAX_WINDOW_BARS = total_bars

    # Square cube state
    growth_idx = 0  # index into GROWTH_CYCLE
    hidden_size = SQUARE_CUBE_SIZES[0]  # start at 4
    H_layers = SQUARE_CUBE_SIZES[0]
    L_layers = SQUARE_CUBE_SIZES[0]
    current_h_dim = hidden_size
    phase = 0

    _validate_square_cube_state(hidden_size, H_layers, L_layers)

    print(f"\nAutoresearch: {total_pool} filtered pairs, {total_bars} bars")
    print(f"Square Cube: hidden_size={hidden_size}, H_layers={H_layers}, L_layers={L_layers}")
    print(f"Bag scaling: 4->5, 16->20, 64->40, 256->80 pairs")
    print(f"Window: [{MIN_WINDOW_BARS}..{MAX_WINDOW_BARS}] bars")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            phase += 1
            progress = min(1.0, phase / 200.0)

            # --- Stochastic bag sampling scaled by current model size ---
            selected_pairs = _stochastic_bag_sample(
                filtered_pairs, current_h_dim, rng,
                min_pairs=MIN_PAIRS, max_pairs=MAX_PAIRS,
            )
            n_pairs = len(selected_pairs)

            window_ceil = int(MIN_WINDOW_BARS + (MAX_WINDOW_BARS - MIN_WINDOW_BARS) * progress)
            window_bars = rng.randint(MIN_WINDOW_BARS, max(MIN_WINDOW_BARS, window_ceil))
            window_bars = min(window_bars, total_bars)

            max_start = max(0, total_bars - window_bars)
            start_bar = rng.randint(0, max_start) if max_start > 0 else 0
            end_bar = start_bar + window_bars

            window_days = round(window_bars * 5 / (60 * 24), 1)

            trial_graph = _make_trial_graph(graph, selected_pairs, start_bar, end_bar)
            if not trial_graph.edges:
                print(f"Phase {phase}: empty graph, skipping")
                continue

            lr = 10 ** random.uniform(-4, -1.5)
            y_depth = random.choice([100, 200, 300, 400])
            x_pixels = random.choice([10, 15, 20, 30])
            curvature = random.uniform(0.5, 4.0)
            prediction_depth = random.choice([1, 2, 3, 5, 10])

            print(f"\n=== Phase {phase} (p={progress:.2f}) ===")
            print(f"  Square: hidden_size={current_h_dim}, H_layers={H_layers}, L_layers={L_layers}")
            print(f"  Bag: {n_pairs} pairs, {window_bars} bars ({window_days}d)")

            model = HierarchicalReasoningModel(
                n_edges=len(trial_graph.edges),
                learning_rate=lr,
                y_depth=y_depth,
                x_pixels=x_pixels,
                curvature=curvature,
                h_dim=current_h_dim,
                z_dim=current_h_dim,
                prediction_depth=prediction_depth,
                H_layers=H_layers,
                L_layers=L_layers,
                H_cycles=2,
                L_cycles=2,
            )
            model.register_edges(list(trial_graph.edges.keys()))

            # Train with loss history tracking
            loss_history = []
            total_loss, n_updates, _, loss_history = run_training(
                trial_graph, model, print_every=1000, loss_history=loss_history
            )

            if n_updates > 0:
                val_bpb = total_loss / n_updates
            else:
                val_bpb = 999.0

            growth_dim = GROWTH_CYCLE[growth_idx]
            print(f"  val_bpb={val_bpb:.6f} (Best: {best_bpb:.6f})")

            if val_bpb < best_bpb:
                best_bpb = val_bpb
                params = {
                    'lr': lr, 'h_dim': current_h_dim,
                    'y_depth': y_depth, 'x_pixels': x_pixels,
                    'curvature': curvature, 'prediction_depth': prediction_depth,
                    'H_layers': H_layers, 'L_layers': L_layers,
                }
                bag_spec = {
                    'n_pairs': n_pairs, 'window_bars': window_bars,
                    'window_days': window_days, 'start_bar': start_bar
                }
                best_params = {**params, **bag_spec, 'growth_phase': f'{growth_dim}'}
                print(f"  --> [NEW BEST] val_bpb: {best_bpb:.6f}")
                model.save("model_weights.pt")

            conn.execute(
                "INSERT INTO experiments (timestamp, val_bpb, params, bag_spec, growth_phase) VALUES (now(), ?, ?, ?, ?)",
                [val_bpb, str(params), str(bag_spec), growth_dim]
            )

            # Check for convergence and trigger growth
            if _is_converged(loss_history):
                old_h = current_h_dim
                old_H = H_layers
                old_L = L_layers

                hidden_size, H_layers, L_layers, did_grow = _apply_growth_step(
                    model, growth_dim, hidden_size, H_layers, L_layers
                )
                current_h_dim = hidden_size

                if did_grow:
                    growth_idx = (growth_idx + 1) % len(GROWTH_CYCLE)
                    print(
                        f"\n  *** CONVERGED -> GROWTH: {growth_dim} "
                        f"[h={old_h}, H={old_H}, L={old_L}] -> "
                        f"[h={current_h_dim}, H={H_layers}, L={L_layers}]"
                    )
                    model.save("model_weights_grown.pt")
                else:
                    print(
                        f"\n  *** CONVERGED -> NO GROWTH: {growth_dim} already at max "
                        f"[h={current_h_dim}, H={H_layers}, L={L_layers}]"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if best_params:
        print(f"Best: val_bpb={best_bpb:.6f} with {best_params}")
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoresearch', action='store_true')
    parser.add_argument('--start-bar', type=int, default=0)
    parser.add_argument('--end-bar', type=int, default=None)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--min-partners', type=int, default=5)
    parser.add_argument('--max-partners', type=int, default=None)
    parser.add_argument('--skip-fetch', action='store_true')
    parser.add_argument('--exchange', type=str, default='coinbase', choices=['coinbase', 'binance'])
    parser.add_argument('--prediction-depth', type=int, default=1)
    parser.add_argument('--h-dim', type=int, default=4)
    parser.add_argument('--z-dim', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--y-depth', type=int, default=200)
    parser.add_argument('--x-pixels', type=int, default=20)
    parser.add_argument('--curvature', type=float, default=2.0)
    args = parser.parse_args()

    print("Loading coin graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load(
        lookback_days=1095 if args.exchange == "binance" else 365,
        min_partners=args.min_partners,
        max_partners=args.max_partners,
        exchange=args.exchange,
        skip_fetch=args.skip_fetch,
    )
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges, {n_bars} bars")
    
    if n_bars == 0:
        print("No data. Run fetch_candles.py first.")
        return

    model = HierarchicalReasoningModel(
        n_edges=len(graph.edges),
        learning_rate=args.lr,
        y_depth=args.y_depth,
        x_pixels=args.x_pixels,
        curvature=args.curvature,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        prediction_depth=args.prediction_depth
    )
    model.register_edges(list(graph.edges.keys()))
    model.load("model_weights.pt")

    db_path = str(Config.DB_PATH)

    if args.autoresearch:
        run_autoresearch(graph, db_path=db_path, exchange=args.exchange)
    else:
        # Stochastic bag sampling for single training run (Binance mode)
        if args.exchange == "binance":
            all_db_pairs = _list_all_binance_pairs(db_path)
            if all_db_pairs:
                filtered = _compute_volatility_filter(
                    db_path, all_db_pairs,
                    lookback_days=1095, granularity="300", min_velocity=0.001,
                )
                print(f"[StochasticBag] Volatility-filtered: {len(filtered)} pairs (from {len(all_db_pairs)})")
                if filtered:
                    rng = random.Random()
                    sampled = _stochastic_bag_sample(
                        filtered, args.h_dim, rng,
                        min_pairs=5, max_pairs=len(filtered),
                    )
                    if sampled:
                        print(f"[StochasticBag] Sampled {len(sampled)} pairs for training")
                        # Rebuild graph with sampled pairs only
                        graph = CoinGraph(fee_rate=0.001)
                        graph.load(
                            lookback_days=1095,
                            min_partners=args.min_partners,
                            max_partners=args.max_partners,
                            exchange=args.exchange,
                            skip_fetch=True,
                        )
                        # Filter graph edges to sampled pairs only
                        sampled_set = set(sampled)
                        pair_to_edges = {}
                        for pid in graph.all_pairs:
                            parts = pid.split("-", 1)
                            if len(parts) == 2:
                                pair_to_edges[pid] = (parts[0], parts[1])
                        from coin_graph import EdgeState, NodeState
                        new_edges = {}
                        new_edge_state = {}
                        new_nodes = set()
                        new_node_state = {}
                        for pid in sampled:
                            if pid not in pair_to_edges:
                                continue
                            base, quote = pair_to_edges[pid]
                            for edge in [(base, quote), (quote, base)]:
                                if edge in graph.edges:
                                    new_edges[edge] = graph.edges[edge]
                                    new_edge_state[edge] = EdgeState()
                            new_nodes.add(base)
                            new_nodes.add(quote)
                            new_node_state.setdefault(base, NodeState())
                            new_node_state.setdefault(quote, NodeState())
                        if new_edges:
                            graph.edges = new_edges
                            graph.edge_state = new_edge_state
                            graph.nodes = new_nodes
                            graph.node_state = new_node_state
                            graph.all_pairs = sampled
                            graph._align_timestamps()
                            n_bars = len(graph.common_timestamps)
                            # Re-register model edges
                            model = HierarchicalReasoningModel(
                                n_edges=len(graph.edges),
                                learning_rate=args.lr,
                                y_depth=args.y_depth,
                                x_pixels=args.x_pixels,
                                curvature=args.curvature,
                                h_dim=args.h_dim,
                                z_dim=args.z_dim,
                                prediction_depth=args.prediction_depth,
                            )
                            model.register_edges(list(graph.edges.keys()))
                            model.load("model_weights.pt")

        loss_history = []
        end_bar = args.end_bar if args.end_bar else min(n_bars, 10000)
        print(f"Training from bar {args.start_bar} to {end_bar}...")

        total_loss, n_updates, _, loss_history = run_training(
            graph, model,
            start_bar=args.start_bar,
            end_bar=end_bar,
            print_every=args.print_every,
            loss_history=loss_history
        )

        avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
        print(f"\nDone: avg_loss={avg_loss:.6f}, n_updates={n_updates}")
        model.save("model_weights.pt")


if __name__ == "__main__":
    main()
