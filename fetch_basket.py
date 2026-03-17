#!/usr/bin/env python3
"""
Fetch top N coins by volume from Coinbase. USD-quoted + cross-pairs for dense graph.

Usage:
    python3 fetch_basket.py --coins 20 --days 365
"""
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from itertools import combinations

import pandas as pd
import duckdb
from coinbase.rest import RESTClient

from config import Config

DB_PATH = "candles.duckdb"
MAX_CANDLES = 300
GRANULARITY = "300"  # 5-minute
API_GRAN = "FIVE_MINUTE"


def get_client():
    return RESTClient(Config.COINBASE_API_KEY, Config.COINBASE_API_SECRET)


def discover_products(client, min_partners=5):
    """
    Build real graph structure from Coinbase products.

    1. Count trading partners for each coin
    2. Select coins with >= min_partners distinct partners
    3. Collect all pairs where BOTH coins meet the threshold
    """
    resp = client.get_public_products()

    # Build adjacency: coin -> set of trading partners
    adjacency = {}
    all_products = {}

    for p in resp.products:
        if p.status != "online" or p.trading_disabled:
            continue

        pid = p.product_id
        vol = float(getattr(p, 'volume_24h', 0) or 0)
        all_products[pid] = vol

        parts = pid.split("-", 1)
        if len(parts) != 2:
            continue

        base, quote = parts

        # Bidirectional adjacency: count distinct partners
        if base not in adjacency:
            adjacency[base] = set()
        if quote not in adjacency:
            adjacency[quote] = set()

        adjacency[base].add(quote)
        adjacency[quote].add(base)

    # Select coins with >= min_partners distinct partners
    selected_coins = []
    for coin in sorted(adjacency.keys()):
        if len(adjacency[coin]) >= min_partners:
            # Find the max volume for any pair involving this coin
            max_vol = 0.0
            for partner in adjacency[coin]:
                for pid in [f"{coin}-{partner}", f"{partner}-{coin}"]:
                    if pid in all_products:
                        max_vol = max(max_vol, all_products[pid])
            selected_coins.append((coin, max_vol, len(adjacency[coin])))

    coin_set = {c for c, _, _ in selected_coins}

    # Collect all pairs where BOTH coins are in coin_set
    pairs_to_fetch = []
    seen_pairs = set()

    for pid, vol in all_products.items():
        parts = pid.split("-", 1)
        if len(parts) != 2:
            continue

        base, quote = parts
        if base in coin_set and quote in coin_set:
            # Avoid duplicates: store canonical form (sorted)
            canonical = tuple(sorted([base, quote]))
            if canonical not in seen_pairs:
                pairs_to_fetch.append((pid, vol))
                seen_pairs.add(canonical)

    pairs_to_fetch.sort(key=lambda x: x[1], reverse=True)
    return selected_coins, pairs_to_fetch


def fetch_candles_range(client, product_id, start_ts, end_ts):
    """Fetch candles between unix timestamps, paginating."""
    candles = []
    current = start_ts
    gran_sec = int(GRANULARITY)
    chunk_size = MAX_CANDLES * gran_sec

    while current < end_ts:
        chunk_end = min(current + chunk_size, end_ts)
        try:
            resp = client.get_public_candles(
                product_id=product_id,
                start=str(current),
                end=str(chunk_end),
                granularity=API_GRAN,
            )
            if hasattr(resp, 'candles') and resp.candles:
                for c in resp.candles:
                    candles.append({
                        'product_id': product_id,
                        'timestamp': datetime.fromtimestamp(int(c.start)),
                        'open': float(c.open),
                        'high': float(c.high),
                        'low': float(c.low),
                        'close': float(c.close),
                        'volume': float(c.volume),
                        'granularity': GRANULARITY,
                    })
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e):
                print(" rate-limited, sleeping 15s...", end="", flush=True)
                time.sleep(15)
                continue  # retry same chunk
            else:
                print(f" err: {e}", end="", flush=True)
        current = chunk_end
        time.sleep(0.5)

    return candles


def save_candles(db, candles):
    if not candles:
        return 0
    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    db.execute("""
        INSERT OR REPLACE INTO candles
        SELECT product_id, timestamp, open, high, low, close, volume, granularity FROM df
    """)
    return len(candles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-partners", type=int, default=5, help="Min trading partners per coin")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    client = get_client()
    db = duckdb.connect(DB_PATH)

    # Ensure table exists
    db.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            product_id VARCHAR, timestamp TIMESTAMP,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, granularity VARCHAR,
            PRIMARY KEY (product_id, timestamp, granularity)
        )
    """)

    print(f"Building graph with coins having >= {args.min_partners} trading partners...")
    selected_coins, pairs = discover_products(client, args.min_partners)

    print(f"\nSelected {len(selected_coins)} coins with >= {args.min_partners} partners:")
    for coin, vol, n_partners in sorted(selected_coins, key=lambda x: -x[2])[:20]:
        print(f"  {coin:>6s}: {n_partners:2d} partners, max_vol=${vol/1e6:,.1f}M")
    if len(selected_coins) > 20:
        print(f"  ... and {len(selected_coins) - 20} more")

    print(f"\n{len(pairs)} unique trading pairs found:")
    for pid, vol in pairs[:30]:
        print(f"  {pid:>12s}: ${vol/1e6:,.1f}M")
    if len(pairs) > 30:
        print(f"  ... and {len(pairs)-30} more")

    # Check existing coverage
    now = datetime.now()
    target_start = now - timedelta(days=args.days)
    start_ts = int(target_start.timestamp())
    end_ts = int(now.timestamp())

    total_saved = 0
    for i, (pid, vol) in enumerate(pairs):
        # Check existing
        existing = db.execute(
            "SELECT COUNT(*) FROM candles WHERE product_id = ? AND granularity = ?",
            [pid, GRANULARITY]
        ).fetchone()[0]

        expected = args.days * 24 * 12  # 5-min bars per day
        if existing >= expected * 0.90:
            print(f"[{i+1}/{len(pairs)}] {pid:>12s}: {existing:,} candles (have enough, skip)")
            continue

        print(f"[{i+1}/{len(pairs)}] {pid:>12s}: fetching {args.days}d...", end="", flush=True)
        candles = fetch_candles_range(client, pid, start_ts, end_ts)
        n = save_candles(db, candles)
        total_saved += n
        print(f" {n:,} saved (total: {existing + n:,})")

    db.close()

    n_coins = len(selected_coins)
    print(f"\nDone. {total_saved:,} new candles saved.")
    print(f"Graph: {n_coins} nodes, {len(pairs)} edges")
    print(f"Graph density: {len(pairs)} / {n_coins*(n_coins-1)/2:.0f} possible = {len(pairs)/(n_coins*(n_coins-1)/2)*100:.0f}%")


if __name__ == "__main__":
    main()
