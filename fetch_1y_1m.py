#!/usr/bin/env python3
"""
Production-grade 1-year 1m candle fetcher for Coinbase Advanced.
Fetches ALL available trading pairs, stores in parquet format.

Usage:
    python3 fetch_1y_1m.py              # fetch all pairs
    python3 fetch_1y_1m.py --dry-run    # just list products
    python3 fetch_1y_1m.py BTC-USD     # fetch single pair
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from coinbase.rest import RESTClient

CACHE_DIR = Path(__file__).parent / "candle_cache_1m"
START_DATE = "2024-01-01"
END_DATE = "2025-01-01"
GRANULARITY = "ONE_MINUTE"
MAX_CANDLES_PER_REQUEST = 300  # Coinbase limit
REQUESTS_PER_MINUTE = 10       # Rate limit safety

def get_client():
    """Create Coinbase client (public for now, can add auth later)."""
    return RESTClient()

def get_products():
    """Fetch all available products from Coinbase."""
    client = get_client()
    products = client.get_public_products()
    
    result = []
    for p in products.products:
        if p.status == "online" and not p.trading_disabled:
            if getattr(p, 'cancel_only', False) or getattr(p, 'limit_only', False):
                continue
            if "-USD" in p.product_id or "-USDC" in p.product_id:
                result.append(p.product_id)
    
    return sorted(result)

def fetch_candles_for_product(client, product_id, start, end):
    """Fetch all candles for a product in chunks."""
    candles = []
    current_start = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    
    while current_start < end_dt:
        current_end = current_start + timedelta(minutes=MAX_CANDLES_PER_REQUEST)
        if current_end > end_dt:
            current_end = end_dt
        
        try:
            # Use epoch timestamps (strings required by API)
            resp = client.get_public_candles(
                product_id=product_id,
                start=str(int(current_start.timestamp())),
                end=str(int(current_end.timestamp())),
                granularity=GRANULARITY
            )
            
            if hasattr(resp, 'candles') and resp.candles:
                for c in resp.candles:
                    candles.append({
                        'timestamp': c.start,
                        'open': float(c.open),
                        'high': float(c.high),
                        'low': float(c.low),
                        'close': float(c.close),
                        'volume': float(c.volume)
                    })
            else:
                break
                
        except Exception as e:
            print(f"  Error fetching {product_id} {current_start}: {e}")
            time.sleep(5)
            continue
        
        current_start = current_end
        time.sleep(60 / REQUESTS_PER_MINUTE)
    
    return product_id, candles

def save_to_parquet(product_id, candles):
    """Save candles to parquet file."""
    if not candles:
        return False
    
    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    filename = CACHE_DIR / f"{product_id.replace('-', '_')}_1m.parquet"
    df.to_parquet(filename, index=False, compression='zstd')
    return len(df)

def load_existing(product_id):
    """Load existing candle data if any."""
    filename = CACHE_DIR / f"{product_id.replace('-', '_')}_1m.parquet"
    if filename.exists():
        df = pd.read_parquet(filename)
        return df
    return None

def fetch_product(product_id, dry_run=False):
    """Fetch candles for a single product."""
    print(f"Processing {product_id}...")
    
    existing = load_existing(product_id)
    if existing is not None and len(existing) > 300000:
        print(f"  Already have {len(existing)} candles, skipping")
        return None
    
    client = get_client()
    product_id, candles = fetch_candles_for_product(client, product_id, START_DATE, END_DATE)
    
    if candles:
        count = save_to_parquet(product_id, candles)
        print(f"  Saved {count} candles to parquet")
        return count
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Fetch 1 year 1m candles from Coinbase")
    parser.add_argument("--dry-run", action="store_true", help="Just list products")
    parser.add_argument("--single", type=str, help="Fetch single product (e.g. BTC-USD)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    args = parser.parse_args()
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching {GRANULARITY} candles from {START_DATE} to {END_DATE}")
    print(f"Output: {CACHE_DIR}")
    print()
    
    if args.single:
        products = [args.single]
    else:
        products = get_products()
        print(f"Found {len(products)} products")
        
        if args.dry_run:
            for p in products:
                print(f"  {p}")
            return
    
    total_candles = 0
    failed = []
    
    for product_id in products:
        try:
            count = fetch_product(product_id)
            if count:
                total_candles += count
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(product_id)
    
    print()
    print(f"Total candles fetched: {total_candles:,}")
    print(f"Files saved to: {CACHE_DIR}")
    
    if failed:
        print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
