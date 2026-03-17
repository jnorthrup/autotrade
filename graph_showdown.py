#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from accel_model import AccelModel
from coin_graph import CoinGraph


# deleted DB_PATH


# persistence logic removed


def run_simulation(graph: CoinGraph, accel_model: AccelModel, start_bar: int = 0, 
                   end_bar: Optional[int] = None, print_every: int = 100):
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
    
    total_pnl = 0.0
    n_trades = 0
    
    path_tracking = {}
    quote_tracking = {}
    
    total_loss = 0.0
    n_updates = 0
    
    bar_idx = start_bar
    while bar_idx < end_bar:
        # OPPORTUNITY-DRIVEN LOAD: If we hit the end of current memory, pull a new chunk
        if bar_idx >= len(graph.common_timestamps):
            added = graph.hydrate_increment(days=7)
            if added == 0:
                print(f"Reached end of available history at bar {bar_idx}.")
                break
                
        edge_accels = graph.update(bar_idx)
        
        if bar_idx % 10 == 0:
            hrm_direction = accel_model.high_level_plan(graph)
            graph.integrate_hrm_output(hrm_direction)
        
        if bar_idx >= 8:
            predicted_accels = accel_model.predict(graph)
            # Bind the model's predictions back into the graph's edge states so Dijkstra can see them
            for edge, pred in predicted_accels.items():
                if edge in graph.edge_state:
                    graph.edge_state[edge].accel = pred
        
        target = graph.best_target()
        
        # DEBUG: Dump predictions and routes every 100 bars to see why trades aren't executing
        if bar_idx > 0 and bar_idx % 100 == 0:
            print(f"\n[DEBUG] Bar {bar_idx} Holding: {holding} Target: {target}")
            print(f"[DEBUG] Target Height: {graph.node_state[target].height if target else 0}")
            paths = graph.dijkstra(holding)
            if target in paths:
                print(f"[DEBUG] Path to target: {paths[target]}")
            else:
                print(f"[DEBUG] No path to target {target} found in {paths.keys()}")
            
        trade = graph.next_hop(holding, target)
        
        if trade:
            base, quote = trade
            if bar_idx > 0 and bar_idx % 100 == 0:
                print(f"[DEBUG] Raw Trade Tuple: {trade}")
            ts = graph.common_timestamps[bar_idx]
            
            es = graph.edge_state.get((base, quote))
            if es:
                # We can just use the graph's natively tracked velocity for PnL!
                # The velocity represents the actual price shift (log return) over the current bar.
                # It handles inversion perfectly within CoinGraph.update() natively.
                pnl = es.velocity - graph.fee_rate
                
                # Check absolute values so we don't accidentally execute trades on total zero-volume ticks
                if abs(es.velocity) > 1e-6:
                    total_pnl += pnl
                    n_trades += 1
                    
                    if bar_idx > 0 and bar_idx % 100 == 0:
                        print(f"[DEBUG] TRADE EXECUTED: {base}->{quote} at tick {bar_idx} | PnL: {pnl:.6f}")
                    
                    graph.reinforce(base, quote, pnl)
                    
                    # Track Quote Stats
                    if quote not in quote_tracking:
                        quote_tracking[quote] = {'n_trades': 0, 'pnl': 0.0}
                    quote_tracking[quote]['n_trades'] += 1
                    quote_tracking[quote]['pnl'] += pnl
                    
                # Track Multi-Hop Path Attempt
                paths = graph.dijkstra(holding)
                if target and target in paths and len(paths[target][1]) > 2:
                    full_path = "->".join(paths[target][1])
                    if full_path not in path_tracking:
                        path_tracking[full_path] = {'n_uses': 0, 'pnl': 0.0}
                    path_tracking[full_path]['n_uses'] += 1
                    path_tracking[full_path]['pnl'] += pnl
                
                holding = quote
                
        # Accel model online gradient descent - trains EVERY bar to predict the environment
        loss = accel_model.update(graph, edge_accels)
        if loss is not None:
            total_loss += loss
            n_updates += 1
        
        if bar_idx % print_every == 0 and bar_idx > 0:
            avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
            print(f"Bar {bar_idx}: PnL={total_pnl:.4f}, trades={n_trades}, holding={holding}, loss={avg_loss:.6f}")
            top5 = graph.node_potentials()[:5]
            print(f"  Top nodes: {[(c, f'{h:.4f}') for h, c in top5]}")
        
        bar_idx += 1
    
    return total_pnl, n_trades, path_tracking, quote_tracking


def run_autoresearch(graph: CoinGraph, max_minutes: int = 5):
    import duckdb
    import random
    
    conn = duckdb.connect('candles.duckdb')
    
    best_bpb = float('inf')
    best_params = None
    
    print("\nStarting infinite autoresearch loop. Press Ctrl+C to stop.")
    try:
        while True:
            # Randomly sample ML hyperparameters
            lr = 10 ** random.uniform(-4, -1.5)
            width = random.choice([32, 64, 128, 256])
            
            # time_constant (curvature): 1.1 is very slow shallow decay, 2.0 is sharp fisheye
            time_constant = random.uniform(1.1, 2.0)
            
            # sequence length for history horizon
            seq_len = random.choice([8, 12, 16, 24]) 
            
            params = {'lr': lr, 'width': width, 'seq_len': seq_len, 'time_constant': time_constant}
            print(f"\n--- Autoresearch Epoch ---")
            print(f"Testing: lr={lr:.5f}, width={width}, seq_len={seq_len}, time_constant={time_constant:.3f}")
            
            accel_model = AccelModel(
                n_edges=len(graph.edges),
                sequence_length=params['seq_len'],
                learning_rate=params['lr'],
                hidden_dim=params['width'],
                time_constant=params['time_constant']
            )
            accel_model.register_edges(list(graph.edges.keys()))
            
            pnl, n_trades, _, _ = run_simulation(graph, accel_model, print_every=1000)
            
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
    args = parser.parse_args()
    
    print("Loading coin graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load()
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges, {n_bars} bars")
    
    if n_bars == 0:
        print("No data loaded. Run fetch_candles.py first.")
        return
    
    accel_model = AccelModel(n_edges=len(graph.edges), sequence_length=8, time_constant=1.5)
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
            print_every=args.print_every
        )
        
        print(f"\n=== Final Results ===")
        print(f"Total PnL: {total_pnl:.4f}")
        print(f"Number of trades: {n_trades}")
        if n_trades > 0:
            print(f"Avg PnL per trade: {total_pnl/n_trades:.6f}")
        
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
            print(f"  {p}: {s['n_uses']} uses, {s['pnl']:.4f} PnL")
            
        print("\n=== Quote Routing Efficiency ===")
        sorted_quotes = sorted(quotes.items(), key=lambda x: x[1]['pnl'], reverse=True)
        for q, s in sorted_quotes[:5]:
            print(f"  {q}: {s['n_trades']} trades, {s['pnl']:.4f} aggregated PnL")


if __name__ == "__main__":
    main()
