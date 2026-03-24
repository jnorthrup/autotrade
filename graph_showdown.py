#!/usr/bin/env python3
import os
os.environ['USE_WS_ONLY'] = 'false'  # Disable WS-only mode for historical data loading
import argparse
import json
import math
import random
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from coinbase.websocket import WSClient

from accel_model import AccelModel as PyTorchAccelModel
from coin_graph import CoinGraph, EdgeState, NodeState
from portfolio_manager import PortfolioManager, Position

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
        """Inject closed bars for ALL pairs into graph and persist to DuckDB."""
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
            
            # Persist to DuckDB/Parquet surface if available
            if hasattr(self.graph, 'cache'):
                try:
                    save_df = pd.DataFrame([{
                        'product_id': pid,
                        'timestamp': ts,
                        'open': b["open"], 'high': b["high"],
                        'low': b["low"],  'close': b["close"],
                        'volume': b["volume"], 'granularity': "300",
                    }])
                    self.graph.cache.save_candles(save_df)
                except Exception as e:
                    print(f"[WS] Persist error: {e}")


        # Forward-fill any other pairs that didn't get a tick this bar
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


def run_simulation(graph: CoinGraph, accel_model: PyTorchAccelModel, start_bar: int = 0,
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
    pending_fraction: float = 0.0                    # position size [0,1] for pending_trade
    predicted_accels: Dict = {}
    
    start_time = time.time()
    MAX_LIVE_SECONDS = 300  # 5 minutes

    while end_bar is None or bar_idx < end_bar:
        if ws and (time.time() - start_time > MAX_LIVE_SECONDS):
            print(f"Live mode runtime limit reached ({MAX_LIVE_SECONDS}s). Stopping.")
            break

        if bar_idx >= len(graph.common_timestamps):
            if ws:
                # Live mode: wait for WSCandles to inject a new bar
                with ws._lock:
                    added = ws._new_bars
                    ws._new_bars = 0
                if added == 0:
                    time.sleep(1)
                    continue
            else:
                added = graph.hydrate_increment(days=7)
                if added == 0:
                    print(f"Reached end of available history at bar {bar_idx}.")
                    break

        # 1. REVEAL bar t actual data
        edge_accels, edge_velocities = graph.update(bar_idx)

        # 2. COLLECT PnL for trade decided at bar t-1 (actual bar t return now known)
        if pending_trade is not None:
            base, quote = pending_trade
            es = graph.edge_state.get((base, quote))
            if es:
                gross = es.velocity - graph.fee_rate
                # Scale by the fraction committed at decision time
                rate = pending_fraction * gross
                capital *= (1 + rate)
                n_trades += 1

                # Protection order outcomes
                if gross <= es.stop_loss:
                    # Stop-loss hit: reinforce with doubled penalty, skip to safety
                    graph.reinforce(base, quote, rate * 2)
                elif gross >= es.ptt:
                    # PTT hit: profit taken, reinforce normally
                    graph.reinforce(base, quote, rate)
                else:
                    graph.reinforce(base, quote, rate)

                pm.record_pnl((base, quote), rate)

                if quote not in quote_tracking:
                    quote_tracking[quote] = {'n_trades': 0, 'pnl': 0.0}
                quote_tracking[quote]['n_trades'] += 1
                quote_tracking[quote]['pnl'] += rate
                holding = quote  # always move — no retroactive undo

        # 3. TRAIN on bar t (no future leakage — bar t is now fully known)
        loss = accel_model.update(graph, edge_accels, bar_idx, actual_velocities=edge_velocities)
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
        #
        # Each frame produces: edge, fraction, PTT, stop-loss.
        # Fraction = signal strength above the noise floor, normalized by band width.
        # PTT and stop-loss are per-edge per-bar from rolling volatility + fees.
        target = graph.best_target()
        candidate = graph.next_hop(holding, target)
        pending_trade = None
        pending_fraction = 0.0
        if candidate is not None and predicted_accels:
            pred = predicted_accels.get(candidate, 0)
            es = graph.edge_state.get(candidate)
            if es and pred > es.ptt:
                band = es.ptt - es.stop_loss  # total band width
                signal = pred - es.ptt        # excess above noise floor
                pending_fraction = min(1.0, signal / band) if band > 0 else 1.0
                pending_trade = candidate

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


def _build_pair_adjacency(all_pairs: List[str]) -> Dict[str, List[str]]:
    """Build currency -> [product_id] adjacency from the full bag.
    Pairs sharing a currency are 'related' (same cluster)."""
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
    """Pick n_pairs related pairs via random-walk on the currency adjacency graph.
    Start from a random currency, greedily expand through shared currencies
    so the subset is a connected cluster, not random garbage."""
    if n_pairs >= len(all_pairs):
        return list(all_pairs)

    currencies = list(adj.keys())
    if not currencies:
        return list(all_pairs)[:n_pairs]

    selected = set()
    # Start from a random currency hub
    seed_currency = rng.choice(currencies)
    frontier = [seed_currency]
    visited_currencies = {seed_currency}

    while len(selected) < n_pairs and frontier:
        curr = frontier.pop(0)
        # Add all pairs touching this currency (up to budget)
        candidates = [p for p in adj.get(curr, []) if p not in selected]
        rng.shuffle(candidates)
        for pid in candidates:
            if len(selected) >= n_pairs:
                break
            selected.add(pid)
            # Expand frontier to the other side of this pair
            parts = pid.split("-", 1)
            for c in parts:
                if c not in visited_currencies:
                    visited_currencies.add(c)
                    frontier.append(c)

        # If frontier exhausted but still need more, pick a new random seed
        if not frontier and len(selected) < n_pairs:
            remaining = [c for c in currencies if c not in visited_currencies]
            if remaining:
                new_seed = rng.choice(remaining)
                frontier.append(new_seed)
                visited_currencies.add(new_seed)

    return list(selected)


def _make_trial_graph(full_graph: CoinGraph, selected_pairs: List[str],
                      start_bar: int, end_bar: int) -> CoinGraph:
    """Build a lightweight CoinGraph for one trial from pre-loaded data.
    Shares DataFrames (no copy), just slices timestamps."""
    trial = CoinGraph(fee_rate=full_graph.fee_rate)
    trial.all_pairs = selected_pairs

    # Map product_id -> edges in the full graph
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
                # Match CoinGraph.load() convention: both directions store base/quote from product_id
                trial.edge_state[edge] = EdgeState(base=base, quote=quote)
        trial.nodes.add(base)
        trial.nodes.add(quote)
        trial.node_state.setdefault(base, NodeState(currency=base))
        trial.node_state.setdefault(quote, NodeState(currency=quote))

    # Slice the common_timestamps to the window
    trial.common_timestamps = full_graph.common_timestamps[start_bar:end_bar]
    # Re-index: trial simulation will run from bar 0 to len(trial.common_timestamps)
    return trial


def run_autoresearch(graph: CoinGraph, max_minutes: int = 5, pm_mode: str = 'single_asset'):
    import duckdb
    import random

    conn = duckdb.connect('candles.duckdb')

    _init_leaderboard_tables(conn)
    # Add bag_spec column if missing
    try:
        conn.execute("ALTER TABLE experiments ADD COLUMN bag_spec VARCHAR DEFAULT ''")
    except Exception:
        pass  # column already exists

    best_bpb = float('inf')
    best_params = None

    # Pre-compute pair adjacency for related-pair selection
    adj = _build_pair_adjacency(graph.all_pairs)
    total_bars = len(graph.common_timestamps)
    total_pairs = len(graph.all_pairs)
    rng = random.Random()  # dedicated RNG for bag selection

    # Curriculum bounds
    MIN_PAIRS = max(4, total_pairs // 8)
    MAX_PAIRS = total_pairs
    MIN_WINDOW_BARS = max(200, total_bars // 20)
    MAX_WINDOW_BARS = total_bars

    trial = 0
    print(f"\nStarting autoresearch with stochastic bags.")
    print(f"  Full bag: {total_pairs} pairs, {total_bars} bars")
    print(f"  Curriculum: pairs [{MIN_PAIRS}..{MAX_PAIRS}], window [{MIN_WINDOW_BARS}..{MAX_WINDOW_BARS}] bars")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        while True:
            trial += 1
            # Curriculum progression: smooth ramp from 0 to 1
            progress = min(1.0, trial / 200.0)

            # --- Stochastic bag spec ---
            # Pair count: ramp from MIN to MAX with noise
            pair_ceil = int(MIN_PAIRS + (MAX_PAIRS - MIN_PAIRS) * progress)
            n_pairs = rng.randint(MIN_PAIRS, max(MIN_PAIRS, pair_ceil))
            selected_pairs = _select_related_pairs(graph.all_pairs, adj, n_pairs, rng)
            n_pairs = len(selected_pairs)  # actual selected (may be fewer if bag is small)

            # Window size: ramp from MIN to MAX with noise
            window_ceil = int(MIN_WINDOW_BARS + (MAX_WINDOW_BARS - MIN_WINDOW_BARS) * progress)
            window_bars = rng.randint(MIN_WINDOW_BARS, max(MIN_WINDOW_BARS, window_ceil))
            window_bars = min(window_bars, total_bars)

            # Stochastic start within available range
            max_start = max(0, total_bars - window_bars)
            start_bar = rng.randint(0, max_start) if max_start > 0 else 0
            end_bar = start_bar + window_bars

            window_days = round(window_bars * 5 / (60 * 24), 1)  # 5-min bars -> days

            # Build trial graph from pre-loaded data (no fetches)
            trial_graph = _make_trial_graph(graph, selected_pairs, start_bar, end_bar)
            if not trial_graph.edges:
                print(f"Trial {trial}: empty graph, skipping")
                continue

            # --- Stochastic hyperparams (unchanged) ---
            lr = 10 ** random.uniform(-4, -1.5)
            width = random.choice([32, 64, 128, 256])
            y_depth = random.choice([100, 200, 300, 400])
            x_pixels = random.choice([10, 15, 20, 30])
            curvature = random.uniform(0.5, 4.0)

            params = {'lr': lr, 'width': width, 'y_depth': y_depth,
                      'x_pixels': x_pixels, 'curvature': curvature}
            bag_spec = {'n_pairs': n_pairs, 'window_bars': window_bars,
                        'window_days': window_days, 'start_bar': start_bar,
                        'pairs': [p for p in selected_pairs[:8]]}  # first 8 for logging

            print(f"\n--- Trial {trial} (progress={progress:.2f}) ---")
            print(f"  Bag: {n_pairs} pairs, {window_bars} bars ({window_days}d), start={start_bar}")
            print(f"  Hyper: lr={lr:.5f}, width={width}, y_depth={y_depth}, x_pixels={x_pixels}, curvature={curvature:.3f}")

            accel_model = PyTorchAccelModel(
                n_edges=len(trial_graph.edges),
                sequence_length=params['x_pixels'],
                learning_rate=params['lr'],
                y_depth=params['y_depth'],
                x_pixels=params['x_pixels'],
                curvature=params['curvature']
            )
            accel_model.register_edges(list(trial_graph.edges.keys()))

            pnl, n_trades, _, _ = run_simulation(
                trial_graph, accel_model, print_every=1000, pm_mode=pm_mode
            )

            if n_trades > 0:
                val_bpb = -pnl / n_trades
            else:
                val_bpb = 999.0

            if val_bpb < best_bpb:
                best_bpb = val_bpb
                best_params = {**params, **bag_spec}
                print(f"  --> [NEW BEST] val_bpb: {best_bpb:.6f}")
            else:
                print(f"  val_bpb: {val_bpb:.6f} (Best: {best_bpb:.6f})")

            # Persist to DuckDB with bag spec
            conn.execute(
                "INSERT INTO experiments (timestamp, val_bpb, params, bag_spec) VALUES (now(), ?, ?, ?)",
                [val_bpb, str(params), str(bag_spec)]
            )

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
    parser.add_argument('--no-live', dest='live', action='store_false', help='Disable live mode, use historical data only')
    parser.add_argument('--refresh-bag', action='store_true', help='Re-discover pairs and update bag.json')
    parser.add_argument('--min-partners', type=int, default=5, help='Minimum connections to include a coin')
    parser.add_argument('--max-partners', type=int, default=None, help='Maximum connections to include a coin')
    parser.add_argument('--ane', action='store_true', help='Use ANE-accelerated model (MLX) instead of PyTorch')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip network fetches, use cached data only')
    parser.add_argument('--exchange', type=str, default='coinbase', choices=['coinbase', 'binance'],
                        help='Data source: coinbase (live/WS) or binance (archive CSV->DuckDB)')
    args = parser.parse_args()

    print("Loading coin graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load(
        lookback_days=1 if args.live else (1095 if args.exchange == "binance" else 365),
        refresh_bag=args.refresh_bag,
        min_partners=args.min_partners,
        max_partners=args.max_partners,
        exchange=args.exchange,
        skip_fetch=args.skip_fetch,
    )
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges, {n_bars} bars")
    
    if n_bars == 0:
        print("No data loaded. Run fetch_candles.py first.")
        return
    
    if args.ane:
        from src.autotrade.ane_model import AccelModel
        print("Using ANE-accelerated model (MLX)")
    else:
        AccelModel = PyTorchAccelModel
        print("Using PyTorch model")
    
    accel_model = AccelModel(n_edges=len(graph.edges), sequence_length=8, y_depth=200, x_pixels=20, curvature=2.0)
    accel_model.register_edges(list(graph.edges.keys()))
    accel_model.load("model_weights.pt")

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
