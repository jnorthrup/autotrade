#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from accel_model import AccelModel
from coin_graph import CoinGraph
from portfolio_manager import PortfolioManager, Position


# deleted DB_PATH


# persistence logic removed


def run_simulation(graph: CoinGraph, accel_model: AccelModel, start_bar: int = 0, 
                   end_bar: Optional[int] = None, print_every: int = 100,
                   pm_mode: str = 'single_asset', initial_capital: float = 10000.0):
    if end_bar is None:
        end_bar = len(graph.common_timestamps)
    
    holding = "USD"
    if holding not in graph.nodes:
        if "USDC" in graph.nodes:
            holding = "USDC"
        elif "USDT" in graph.nodes:
            holding = "USDT"
        else:
            holding = list(graph.nodes)[0]

    capital = initial_capital
    n_trades = 0

    path_tracking = {}
    quote_tracking = {}

    total_loss = 0.0
    n_updates = 0

    pm = PortfolioManager(
        mode=pm_mode,
        initial_capital=initial_capital,
        fee_rate=graph.fee_rate
    )
    
    bar_idx = start_bar
    pending_trade: Optional[Tuple[str, str]] = None  # decided at bar t, collected at bar t+1
    predicted_accels: Dict = {}

    while bar_idx < end_bar:
        if bar_idx >= len(graph.common_timestamps):
            added = graph.hydrate_increment(days=7)
            if added == 0:
                print(f"Reached end of available history at bar {bar_idx}.")
                break

        # 1. REVEAL bar t actual data
        edge_accels = graph.update(bar_idx)

        # 2. COLLECT PnL for trade decided at bar t-1 (actual bar t return now known)
        if pending_trade is not None:
            base, quote = pending_trade
            es = graph.edge_state.get((base, quote))
            if es:
                rate = es.velocity - graph.fee_rate  # always collect — positive or negative
                capital *= (1 + rate)
                n_trades += 1
                graph.reinforce(base, quote, rate)
                pm.record_pnl((base, quote), rate)

                if quote not in quote_tracking:
                    quote_tracking[quote] = {'n_trades': 0, 'pnl': 0.0}
                quote_tracking[quote]['n_trades'] += 1
                quote_tracking[quote]['pnl'] += rate
                holding = quote  # always move — no retroactive undo

        # 3. TRAIN on bar t (no future leakage — bar t is now fully known)
        loss = accel_model.update(graph, edge_accels, bar_idx)
        if loss is not None:
            total_loss += loss
            n_updates += 1

        # 4. PREDICT bar t+1 using model trained through bar t
        if bar_idx % 10 == 0:
            hrm_direction = accel_model.high_level_plan(graph)
            graph.integrate_hrm_output(hrm_direction)

        if bar_idx >= 8:
            predicted_accels = accel_model.predict(graph)
            for edge, pred in predicted_accels.items():
                if edge in graph.edge_state:
                    graph.edge_state[edge].accel = pred
            graph._compute_heights(predicted_accels)

        # 5. DECIDE trade for bar t+1 based solely on predictions
        target = graph.best_target()
        candidate = graph.next_hop(holding, target)
        if candidate is not None and predicted_accels.get(candidate, 0) > 0:
            pending_trade = candidate
        else:
            pending_trade = None  # no positive signal — hold

        if pending_trade is not None:
            paths = graph.dijkstra(holding)
            if target and target in paths and len(paths[target][1]) > 2:
                full_path = "->".join(paths[target][1])
                if full_path not in path_tracking:
                    path_tracking[full_path] = {'n_uses': 0, 'pnl': 0.0}
                path_tracking[full_path]['n_uses'] += 1

        if capital < initial_capital * 0.05:
            print(f"Ruin floor hit at bar {bar_idx}: capital=${capital:.2f}. Stopping.")
            break

        if bar_idx % print_every == 0 and bar_idx > 0:
            avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
            print(f"Bar {bar_idx}: capital=${capital:.2f} ({capital-initial_capital:+.2f}), trades={n_trades}, loss={avg_loss:.6f}")
            top5 = graph.node_potentials()[:5]
            print(f"  Top nodes: {[(c, f'{h:.4f}') for h, c in top5]}")

        bar_idx += 1
    
    return capital, n_trades, path_tracking, quote_tracking


def run_autoresearch(graph: CoinGraph, max_minutes: int = 5, pm_mode: str = 'single_asset'):
    import duckdb
    import random
    
    conn = duckdb.connect('candles.duckdb')
    
    best_bpb = float('inf')
    best_params = None
    
    print("\nStarting infinite autoresearch loop. Press Ctrl+C to stop.")
    try:
        while True:
            lr = 10 ** random.uniform(-4, -1.5)
            width = random.choice([32, 64, 128, 256])
            y_depth = random.choice([100, 200, 300, 400])
            x_pixels = random.choice([10, 15, 20, 30])
            curvature = random.uniform(0.5, 4.0)
            
            params = {'lr': lr, 'width': width, 'y_depth': y_depth, 'x_pixels': x_pixels, 'curvature': curvature}
            print(f"\n--- Autoresearch Epoch ---")
            print(f"Testing: lr={lr:.5f}, width={width}, y_depth={y_depth}, x_pixels={x_pixels}, curvature={curvature:.3f}")
            
            accel_model = AccelModel(
                n_edges=len(graph.edges),
                sequence_length=params['x_pixels'],
                learning_rate=params['lr'],
                hidden_dim=params['width'],
                y_depth=params['y_depth'],
                x_pixels=params['x_pixels'],
                curvature=params['curvature']
            )
            accel_model.register_edges(list(graph.edges.keys()))
            
            pnl, n_trades, _, _ = run_simulation(graph, accel_model, print_every=1000, pm_mode=pm_mode)
            
            if n_trades > 0:
                val_bpb = -pnl / n_trades
            else:
                val_bpb = 999.0
            
            status = "completed"
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                best_params = params
                print(f"--> [NEW BEST] val_bpb: {best_bpb:.6f}")
            else:
                print(f"Epoch val_bpb: {val_bpb:.6f} (Best: {best_bpb:.6f})")
            
            # Persist directly to DuckDB
            conn.execute("INSERT INTO experiments (timestamp, val_bpb, params) VALUES (now(), ?, ?)", 
                         [val_bpb, str(params)])
            
    except KeyboardInterrupt:
        print("\nInterrupted autoresearch.")
        
    if best_params:
        print(f"Overall Best: val_bpb={best_bpb:.6f} with {best_params}")
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoresearch', action='store_true')
    parser.add_argument('--start-bar', type=int, default=0)
    parser.add_argument('--end-bar', type=int, default=None)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--pm-mode', type=str, default='single_asset', 
                        choices=['single_asset', 'fractional', 'multi_asset'],
                        help='Portfolio manager mode: single_asset (one trade), fractional (Kelly), multi_asset (rank all flows)')
    parser.add_argument('--initial-capital', type=float, default=100.0)
    args = parser.parse_args()
    
    print("Loading coin graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load()
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges, {n_bars} bars")
    
    if n_bars == 0:
        print("No data loaded. Run fetch_candles.py first.")
        return
    
    accel_model = AccelModel(n_edges=len(graph.edges), sequence_length=8, y_depth=200, x_pixels=20, curvature=2.0)
    accel_model.register_edges(list(graph.edges.keys()))
    
    if args.autoresearch:
        run_autoresearch(graph)
    else:
        end_bar = args.end_bar if args.end_bar else min(n_bars, 10000)
        print(f"Running simulation from bar {args.start_bar} to {end_bar}...")
        
        total_pnl, n_trades, paths, quotes = run_simulation(
            graph, accel_model, 
            start_bar=args.start_bar, 
            end_bar=end_bar,
            print_every=args.print_every,
            pm_mode=args.pm_mode,
            initial_capital=args.initial_capital
        )
        
        print(f"\n=== Final Results ===")
        print(f"Start: ${args.initial_capital:.2f}  End: ${total_pnl:.2f}  Gain: {(total_pnl/args.initial_capital - 1)*100:+.2f}%")
        print(f"Number of trades: {n_trades}")
        if n_trades > 0:
            print(f"Avg gain per trade: {(total_pnl/args.initial_capital - 1)*100/n_trades:+.4f}%")
        
        print("\n=== Node Potentials (Top 10) ===")
        for h, c in graph.node_potentials()[:10]:
            print(f"  {c}: {h:.6f}")
        
        print("\n=== Edge Stats (Top 10 by conductance) ===")
        edge_stats = sorted(graph.get_edge_stats(), key=lambda x: x['conductance'], reverse=True)
        for es in edge_stats[:10]:
            print(f"  {es['edge']}: conductance={es['conductance']:.3f}, "
                  f"cumulative_pnl={es['cumulative_pnl']:.4f}, traversals={es['n_traversals']}")
                  
        print("\n=== Multi-Hop Path Stats ===")
        sorted_paths = sorted(paths.items(), key=lambda x: x[1]['pnl'], reverse=True)
        for p, s in sorted_paths[:5]:
            print(f"  {p}: {s['n_uses']} uses, {s['pnl']*100:+.3f}% total rate")

        print("\n=== Quote Routing Efficiency ===")
        sorted_quotes = sorted(quotes.items(), key=lambda x: x[1]['pnl'], reverse=True)
        for q, s in sorted_quotes[:5]:
            print(f"  {q}: {s['n_trades']} trades, {s['pnl']*100:+.3f}% total rate")


if __name__ == "__main__":
    main()
