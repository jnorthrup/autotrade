#!/usr/bin/env python3
import argparse
import json
import math
import random
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from coinbase.websocket import WSClient

from .accel_model import AccelModel
from .coin_graph import CoinGraph
from .portfolio_manager import PortfolioManager, Position

BAR_SECONDS = 300  # 5-minute bars


class WSCandles:
    """
    Aggregates live ticks into 5-min OHLCV bars.
    When a bar closes it's injected into graph.edges and graph.common_timestamps
    so run_simulation picks it up on the next hydrate check.
    """
    def __init__(self, graph: CoinGraph):
        self.graph = graph
        self._bars: Dict[str, dict] = {}   # product_id -> open bar
        self._lock = threading.Lock()
        self._new_bars = 0
        self._client: Optional[WSClient] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _bar_start(self, ts: float) -> float:
        return ts - (ts % BAR_SECONDS)

    def _on_message(self, msg):
        try:
            data = json.loads(msg) if isinstance(msg, str) else msg
        except Exception:
            return
        for event in data.get("events", []):
            for ticker in event.get("tickers", []):
                pid = ticker.get("product_id", "")
                price_str = ticker.get("price")
                if not pid or not price_str:
                    continue
                try:
                    price = float(price_str)
                    volume = float(ticker.get("volume_24_h", 0) or 0)
                except ValueError:
                    continue
                if price <= 0:
                    continue
                now = time.time()
                bs = self._bar_start(now)
                with self._lock:
                    if pid not in self._bars or self._bars[pid]["start"] < bs:
                        # Close previous bar
                        if pid in self._bars:
                            self._flush_bar(pid)
                        self._bars[pid] = {"start": bs, "open": price,
                                           "high": price, "low": price,
                                           "close": price, "volume": 0.0}
                    b = self._bars[pid]
                    b["high"] = max(b["high"], price)
                    b["low"] = min(b["low"], price)
                    b["close"] = price
                    b["volume"] += volume

    def _flush_bar(self, pid: str):
        """Inject closed bars for ALL pairs into graph to keep them aligned."""
        b = self._bars[pid]
        bs_float = b["start"]
        ts = pd.Timestamp(datetime.fromtimestamp(bs_float, tz=timezone.utc).replace(tzinfo=None))
        
        # Inject the actual closed bar for this PID
        parts = pid.split("-", 1)
        if len(parts) == 2:
            base, quote = parts
            row = pd.DataFrame([{
                "open": b["open"], "high": b["high"],
                "low": b["low"],  "close": b["close"],
                "volume": b["volume"], "granularity": "300",
            }], index=[ts])
            
            for edge in [(base, quote), (quote, base)]:
                if edge in self.graph.edges:
                    df = self.graph.edges[edge]
                    if ts not in df.index:
                        self.graph.edges[edge] = pd.concat([df, row])

        # Forward-fill any other pairs that didn't get a tick this bar
        # This prevents the graph from having 'missing' nodes at the new timestamp
        for other_pid in self._product_ids:
            if other_pid == pid:
                continue
            
            other_parts = other_pid.split("-", 1)
            if len(other_parts) != 2:
                continue
            
            o_base, o_quote = other_parts
            edge = (o_base, o_quote)
            if edge not in self.graph.edges:
                continue
            
            df = self.graph.edges[edge]
            if ts not in df.index:
                # Use last known close as the OHLC for this empty bar
                last_val = df['close'].iloc[-1] if len(df) > 0 else 0.0
                fill_row = pd.DataFrame([{
                    "open": last_val, "high": last_val,
                    "low": last_val,  "close": last_val,
                    "volume": 0.0, "granularity": "300",
                }], index=[ts])
                
                for e in [(o_base, o_quote), (o_quote, o_base)]:
                    self.graph.edges[e] = pd.concat([self.graph.edges[e], fill_row])

        if ts not in self.graph.common_timestamps:
            self.graph.common_timestamps.append(ts)
            self.graph.common_timestamps.sort()
            self._new_bars += 1
            print(f"[WS] Bar aligned for ALL pairs at {ts}")

    def start(self):
        self._product_ids = list(self.graph.all_pairs)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[WS] subscribing {len(self._product_ids)} pairs...")

    def stop(self):
        self._stop.set()
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                self._client = WSClient(api_key=None, api_secret=None,
                                        on_message=self._on_message)
                self._client.open()
                self._client.subscribe(self._product_ids, ["ticker"])
                self._client.run_forever_with_exception_check()
            except Exception as e:
                if self._stop.is_set():
                    break
                print(f"[WS] {e}, reconnect in 5s...")
                time.sleep(5)


def run_simulation(graph: CoinGraph, accel_model: AccelModel, start_bar: int = 0,
                   end_bar: Optional[int] = None, print_every: int = 100,
                   pm_mode: str = 'single_asset', initial_capital: float = 10000.0,
                   ws: Optional[WSCandles] = None):
    if end_bar is None and ws is None:
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

    while end_bar is None or bar_idx < end_bar:
        if bar_idx >= len(graph.common_timestamps):
            if ws:
                with ws._lock:
                    added = ws._new_bars
                    ws._new_bars = 0
                if added == 0:
                    time.sleep(1)
                    continue
            else:
                added = graph.hydrate_increment(days=7)
                if added == 0:
                    break

        # 1. REVEAL bar t actual data
        edge_accels = graph.update(bar_idx)

        # 2. COLLECT PnL for trade decided at bar t-1 (actual bar t return now known)
        if pending_trade is not None:
            base, quote = pending_trade
            es = graph.edge_state.get((base, quote))
            if es:
                rate = es.velocity - graph.fee_rate
                capital *= (1 + rate)
                n_trades += 1
                graph.reinforce(base, quote, rate)
                pm.record_pnl((base, quote), rate)
                quote_tracking.setdefault(quote, {'n_trades': 0, 'pnl': 0.0})
                quote_tracking[quote]['n_trades'] += 1
                quote_tracking[quote]['pnl'] += rate
                holding = quote

        # 3. TRAIN on bar t completed horizons (no future leakage)
        if bar_idx > 20:
            actual_horizons = {}
            for edge in graph.edge_names:
                df = graph.edges.get(edge)
                if df is None: continue
                h_rets = []
                for h in [1, 3, 5, 10, 20]:
                    # sum of log returns for window ending at bar_idx
                    window = df.iloc[max(0, bar_idx-h):bar_idx]
                    if not window.empty:
                        ret = np.log(window['close'].iloc[-1] / window['open'].iloc[0])
                        if graph.edge_state[edge].base != edge[0]: ret *= -1
                        h_rets.append(ret)
                    else: h_rets.append(0.0)
                actual_horizons[edge] = np.array(h_rets)
            
            loss = accel_model.update(graph, actual_horizons, bar_idx)
            if loss is not None:
                total_loss += loss
                n_updates += 1

        # 4. PREDICT bar t+1 using pure weight-driven HRM
        if bar_idx >= 8:
            # Multi-horizon prediction: [1, 3, 5, 10, 20]
            predicted_horizons = accel_model.predict(graph)
            
            # Inject agreement signal into graph potentials
            for edge, h_preds in predicted_horizons.items():
                if edge in graph.edge_state:
                    # Signal = intensity mean across horizons
                    graph.edge_state[edge].accel = np.mean(h_preds)
            
            # Recompute Dijkstra potentials based on model gravity
            graph._compute_heights({e: np.mean(h) for e, h in predicted_horizons.items()})

        # 5. DECIDE for bar t+1
        target = graph.best_target()
        candidate = None
        if target and holding != target:
            paths = graph.dijkstra(holding)
            if target in paths and len(paths[target][1]) >= 2:
                path = paths[target][1]
                candidate = (path[0], path[1])

        # Portfolio Manager handles 'Narrowing Threshold' consistency logic
        # If signal across horizons isn't intense and agreed, decision is None
        decision = pm.decide(graph, holding, candidate, predicted_horizons if bar_idx >= 8 else {})
        if decision:
            pending_trade = (decision.base, decision.quote)
            path_str = f"{decision.base}->{decision.quote}"
            path_tracking.setdefault(path_str, {'n_uses': 0, 'pnl': 0.0})
            path_tracking[path_str]['n_uses'] += 1
        else:
            pending_trade = None

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
    
    _init_leaderboard_tables(conn)
    
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


def _init_leaderboard_tables(conn):
    """Initialize DuckDB tables for leaderboard stats."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            timestamp TIMESTAMP DEFAULT now(),
            val_bpb DOUBLE,
            params VARCHAR
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_stats (
            run_id VARCHAR,
            timestamp TIMESTAMP DEFAULT now(),
            currency VARCHAR,
            avg_height DOUBLE,
            time_at_north_pct DOUBLE,
            net_inflow DOUBLE,
            total_bars INTEGER
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_stats (
            run_id VARCHAR,
            timestamp TIMESTAMP DEFAULT now(),
            edge VARCHAR,
            conductance DOUBLE,
            cumulative_pnl DOUBLE,
            n_traversals INTEGER,
            avg_accel DOUBLE,
            last_pnl DOUBLE
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS path_stats (
            run_id VARCHAR,
            timestamp TIMESTAMP DEFAULT now(),
            path VARCHAR,
            n_uses INTEGER,
            total_pnl DOUBLE
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quote_routing_stats (
            run_id VARCHAR,
            timestamp TIMESTAMP DEFAULT now(),
            quote_currency VARCHAR,
            n_trades INTEGER,
            total_pnl DOUBLE
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hrm_stats (
            run_id VARCHAR,
            timestamp TIMESTAMP DEFAULT now(),
            currency VARCHAR,
            high_level_direction VARCHAR,
            low_level_accuracy DOUBLE
        )
    """)


def persist_leaderboard_stats(graph, accel_model, path_tracking, quote_tracking, run_id: str, conn):
    """Persist leaderboard stats to DuckDB."""
    import uuid
    
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    node_stats = graph.get_node_stats()
    for ns in node_stats:
        conn.execute("""
            INSERT INTO node_stats (run_id, currency, avg_height, time_at_north_pct, net_inflow, total_bars)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [run_id, ns['currency'], ns['avg_height'], ns['time_at_north_pct'], ns['net_inflow'], ns['total_bars']])
    
    edge_stats = graph.get_edge_stats()
    for es in edge_stats:
        conn.execute("""
            INSERT INTO edge_stats (run_id, edge, conductance, cumulative_pnl, n_traversals, avg_accel, last_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [run_id, es['edge'], es['conductance'], es['cumulative_pnl'], es['n_traversals'], es['avg_accel'], es['last_pnl']])
    
    for path, stats in path_tracking.items():
        conn.execute("""
            INSERT INTO path_stats (run_id, path, n_uses, total_pnl)
            VALUES (?, ?, ?, ?)
        """, [run_id, path, stats['n_uses'], stats['pnl']])
    
    for quote, stats in quote_tracking.items():
        conn.execute("""
            INSERT INTO quote_routing_stats (run_id, quote_currency, n_trades, total_pnl)
            VALUES (?, ?, ?, ?)
        """, [run_id, quote, stats['n_trades'], stats['pnl']])
    
    hrm_stats = accel_model.get_hrm_stats()
    for hs in hrm_stats:
        conn.execute("""
            INSERT INTO hrm_stats (run_id, currency, high_level_direction, low_level_accuracy)
            VALUES (?, ?, ?, ?)
        """, [run_id, hs['currency'], hs['high_level_direction'], hs['low_level_accuracy']])
    
    return run_id


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
    parser.add_argument('--live', action='store_true', help='Live mode: WS feeds bars after history exhausted')
    parser.add_argument('--refresh-bag', action='store_true', help='Re-discover pairs and update bag.json')
    parser.add_argument('--min-partners', type=int, default=5, help='Minimum connections to include a coin')
    parser.add_argument('--max-partners', type=int, default=None, help='Maximum connections to include a coin')
    args = parser.parse_args()
    
    print("Loading coin graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load(
        lookback_days=1 if args.live else 365, 
        refresh_bag=args.refresh_bag,
        min_partners=args.min_partners,
        max_partners=args.max_partners
    )
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges, {n_bars} bars")
    
    if n_bars == 0:
        print("No data loaded. Run fetch_candles.py first.")
        return
    
    accel_model = AccelModel(n_edges=len(graph.edges), sequence_length=8, y_depth=200, x_pixels=20, curvature=2.0)
    accel_model.register_edges(list(graph.edges.keys()))

    ws = None
    if args.live:
        ws = WSCandles(graph)
        ws.start()

    if args.autoresearch:
        run_autoresearch(graph)
    else:
        end_bar = None if args.live else (args.end_bar if args.end_bar else min(n_bars, 10000))
        print(f"Running simulation from bar {args.start_bar} to {end_bar}...")
        
        total_pnl, n_trades, paths, quotes = run_simulation(
            graph, accel_model,
            start_bar=args.start_bar,
            end_bar=end_bar,
            print_every=args.print_every,
            pm_mode=args.pm_mode,
            initial_capital=args.initial_capital,
            ws=ws,
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
