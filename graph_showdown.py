#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    
    total_pnl = 0.0
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
        else:
            predicted_accels = {}
        
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
        
        if pm_mode == 'multi_asset':
            decisions = pm.rank_all_flows(graph, predicted_accels)
            
            if bar_idx > 0 and bar_idx % 100 == 0:
                print(f"\n[MULTI] Bar {bar_idx}: {len(decisions)} flows ranked")
            
            for decision in decisions[:10]:
                base, quote = decision.base, decision.quote
                edge = (base, quote)
                es = graph.edge_state.get(edge)
                if es and abs(es.velocity) > 1e-6:
                    trade_value = abs(es.velocity)
                    pnl = es.velocity - graph.fee_rate
                    adjusted_pnl = pnl * decision.fraction
                    total_pnl += adjusted_pnl
                    n_trades += 1
                    graph.reinforce(base, quote, adjusted_pnl)
                    pm.record_pnl(edge, adjusted_pnl)
                    
                    if base not in pm.positions:
                        pm.positions[base] = Position(currency=base, size=0.0, entry_value=1.0)
                    pm.positions[base].size += trade_value * decision.fraction
                    
                    if bar_idx > 0 and bar_idx % 100 == 0:
                        print(f"  {base}->{quote}: kelly={decision.kelly_fraction:.2f} conf={decision.confidence:.2f} pnl={adjusted_pnl:.4f}")
        else:
            decision = pm.decide(graph, holding, trade, predicted_accels)
            
            if decision and decision.fraction > 0:
                base, quote = trade
                
                if bar_idx > 0 and bar_idx % 100 == 0:
                    print(f"[DEBUG] Trade Decision: {base}->{quote} fraction={decision.fraction:.2f} kelly={decision.kelly_fraction:.2f} regime={decision.regime}")
                
                es = graph.edge_state.get((base, quote))
                if es:
                    pnl = es.velocity - graph.fee_rate
                    adjusted_pnl = pnl * decision.fraction
                    
                    if abs(es.velocity) > 1e-6:
                        total_pnl += adjusted_pnl
                        n_trades += 1
                        
                        if bar_idx > 0 and bar_idx % 100 == 0:
                            print(f"[DEBUG] TRADE EXECUTED: {base}->{quote} at tick {bar_idx} | PnL: {adjusted_pnl:.6f} (fraction: {decision.fraction})")
                        
                        graph.reinforce(base, quote, adjusted_pnl)
                        pm.record_pnl((base, quote), adjusted_pnl)
                    
                    if quote not in quote_tracking:
                        quote_tracking[quote] = {'n_trades': 0, 'pnl': 0.0}
                    quote_tracking[quote]['n_trades'] += 1
                    quote_tracking[quote]['pnl'] += adjusted_pnl
                    
                    paths = graph.dijkstra(holding)
                    if target and target in paths and len(paths[target][1]) > 2:
                        full_path = "->".join(paths[target][1])
                        if full_path not in path_tracking:
                            path_tracking[full_path] = {'n_uses': 0, 'pnl': 0.0}
                        path_tracking[full_path]['n_uses'] += 1
                        path_tracking[full_path]['pnl'] += adjusted_pnl
                    
                    holding = quote
                
        # Accel model online gradient descent - trains EVERY bar to predict the environment
        loss = accel_model.update(graph, edge_accels, bar_idx)
        if loss is not None:
            total_loss += loss
            n_updates += 1
        
        if bar_idx % print_every == 0 and bar_idx > 0:
            avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
            pm_stats = pm.get_stats(graph)
            usd_value = pm_stats.get('total_value_usd', pm_stats['total_value'])
            print(f"Bar {bar_idx}: PnL={total_pnl:.4f}, trades={n_trades}, USD={usd_value:.2f}, loss={avg_loss:.6f}")
            top5 = graph.node_potentials()[:5]
            print(f"  Top nodes: {[(c, f'{h:.4f}') for h, c in top5]}")
        
        bar_idx += 1
    
    return total_pnl, n_trades, path_tracking, quote_tracking


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
    parser.add_argument('--initial-capital', type=float, default=10000.0)
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
