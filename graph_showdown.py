#!/usr/bin/env python3
import os
os.environ['USE_WS_ONLY'] = 'false'
import argparse
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from accel_model import AccelModel
from coin_graph import CoinGraph
from candle_cache import CandleCache

import duckdb


def run_training(graph: CoinGraph, model: AccelModel, start_bar: int = 0,
                 end_bar: Optional[int] = None, print_every: int = 100) -> Tuple[float, int, bool]:
    if end_bar is None:
        end_bar = len(graph.common_timestamps)
    
    total_loss = 0.0
    n_updates = 0
    early_stopped = False
    
    for bar_idx in range(start_bar, end_bar):
        if bar_idx >= len(graph.common_timestamps):
            break
        
        edge_accels, edge_velocities, hit_ptt, hit_stop = graph.update(bar_idx)
        
        if bar_idx >= model.prediction_depth:
            model.predict(graph, bar_idx)
        
        if bar_idx >= model.prediction_depth * 2:
            loss = model.update(graph, edge_accels, bar_idx, hit_ptt=hit_ptt, hit_stop=hit_stop)
            if loss is not None:
                total_loss += loss
                n_updates += 1
        
        if bar_idx % print_every == 0 and bar_idx > 0:
            avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
            print(f"Bar {bar_idx}: avg_loss={avg_loss:.6f}")
    
    return total_loss, n_updates, early_stopped


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
                trial.edge_state[edge] = EdgeState(base=base, quote=quote)
        trial.nodes.add(base)
        trial.nodes.add(quote)
        trial.node_state.setdefault(base, NodeState(currency=base))
        trial.node_state.setdefault(quote, NodeState(currency=quote))

    trial.common_timestamps = full_graph.common_timestamps[start_bar:end_bar]
    return trial


def run_autoresearch(graph: CoinGraph, pm_mode: str = 'single_asset'):
    conn = duckdb.connect('candles.duckdb')
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            timestamp TIMESTAMP DEFAULT now(),
            val_bpb DOUBLE,
            params VARCHAR,
            bag_spec VARCHAR
        )
    """)

    best_bpb = float('inf')
    best_params = None

    adj = _build_pair_adjacency(graph.all_pairs)
    total_bars = len(graph.common_timestamps)
    total_pairs = len(graph.all_pairs)
    rng = random.Random()

    MIN_PAIRS = max(4, total_pairs // 8)
    MAX_PAIRS = total_pairs
    MIN_WINDOW_BARS = max(200, total_bars // 20)
    MAX_WINDOW_BARS = total_bars

    trial = 0
    print(f"\nAutoresearch: {total_pairs} pairs, {total_bars} bars")
    print(f"Curriculum: pairs [{MIN_PAIRS}..{MAX_PAIRS}], window [{MIN_WINDOW_BARS}..{MAX_WINDOW_BARS}]")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            trial += 1
            progress = min(1.0, trial / 200.0)

            pair_ceil = int(MIN_PAIRS + (MAX_PAIRS - MIN_PAIRS) * progress)
            n_pairs = rng.randint(MIN_PAIRS, max(MIN_PAIRS, pair_ceil))
            selected_pairs = _select_related_pairs(graph.all_pairs, adj, n_pairs, rng)
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
                print(f"Trial {trial}: empty graph, skipping")
                continue

            lr = 10 ** random.uniform(-4, -1.5)
            h_dim = random.choice([8, 16, 32])
            z_dim = random.choice([4, 8, 16])
            y_depth = random.choice([100, 200, 300, 400])
            x_pixels = random.choice([10, 15, 20, 30])
            curvature = random.uniform(0.5, 4.0)
            prediction_depth = random.choice([1, 2, 3, 5, 10])

            params = {
                'lr': lr, 'h_dim': h_dim, 'z_dim': z_dim,
                'y_depth': y_depth, 'x_pixels': x_pixels,
                'curvature': curvature, 'prediction_depth': prediction_depth
            }
            bag_spec = {
                'n_pairs': n_pairs, 'window_bars': window_bars,
                'window_days': window_days, 'start_bar': start_bar
            }

            print(f"\n--- Trial {trial} (p={progress:.2f}) ---")
            print(f"  Bag: {n_pairs} pairs, {window_bars} bars ({window_days}d)")
            print(f"  depth={prediction_depth}, z_dim={z_dim}, h_dim={h_dim}, lr={lr:.5f}")

            model = AccelModel(
                n_edges=len(trial_graph.edges),
                learning_rate=lr,
                y_depth=y_depth,
                x_pixels=x_pixels,
                curvature=curvature,
                h_dim=h_dim,
                z_dim=z_dim,
                prediction_depth=prediction_depth
            )
            model.register_edges(list(trial_graph.edges.keys()))

            total_loss, n_updates, _ = run_training(
                trial_graph, model, print_every=1000
            )

            if n_updates > 0:
                val_bpb = total_loss / n_updates
            else:
                val_bpb = 999.0

            if val_bpb < best_bpb:
                best_bpb = val_bpb
                best_params = {**params, **bag_spec}
                print(f"  --> [NEW BEST] val_bpb: {best_bpb:.6f}")
            else:
                print(f"  val_bpb: {val_bpb:.6f} (Best: {best_bpb:.6f})")

            conn.execute(
                "INSERT INTO experiments (timestamp, val_bpb, params, bag_spec) VALUES (now(), ?, ?, ?)",
                [val_bpb, str(params), str(bag_spec)]
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
    parser.add_argument('--h-dim', type=int, default=16)
    parser.add_argument('--z-dim', type=int, default=8)
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

    model = AccelModel(
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

    if args.autoresearch:
        run_autoresearch(graph)
    else:
        end_bar = args.end_bar if args.end_bar else min(n_bars, 10000)
        print(f"Training from bar {args.start_bar} to {end_bar}...")
        
        total_loss, n_updates, _ = run_training(
            graph, model,
            start_bar=args.start_bar,
            end_bar=end_bar,
            print_every=args.print_every
        )
        
        avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
        print(f"\nDone: avg_loss={avg_loss:.6f}, n_updates={n_updates}")
        model.save("model_weights.pt")


if __name__ == "__main__":
    main()
