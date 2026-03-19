#!/usr/bin/env python3
"""
Production candle ingestion system for Coinbase Advanced Cross-Pairs.
Focuses specifically on crypto-quoted pairs (e.g., BTC, ETH) to provide
lateral paths for the Dijkstra algorithm. Uses DuckDB for storage.

Usage:
    python3 fetch_cross_pairs_1m.py
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from candle_cache import CandleCache
from coinbase.rest import RESTClient

DB_PATH = Config.DB_PATH
MAX_CANDLES = 350

class CrossPairIngestor:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key or Config.COINBASE_API_KEY
        self.api_secret = api_secret or Config.COINBASE_API_SECRET
        self.client = RESTClient(self.api_key, self.api_secret)
        self.cache = CandleCache(str(DB_PATH))
        self.db = duckdb.connect(str(DB_PATH))
    
    def get_cross_products(self):
        resp = self.client.get_public_products()
        
        # We only want pairs where the quote is a major crypto 
        # that we care about tracking laterally.
        TARGET_QUOTES = {'BTC', 'ETH', 'SOL', 'AVAX', 'ADA'}
        
        products = []
        for p in resp.products:
            if p.status != "online" or p.trading_disabled:
                continue
                
            quote = p.quote_currency_id.upper()
            
            # Key difference: ONLY keep it if it's in TARGET_QUOTES
            if quote in TARGET_QUOTES:
                products.append({
                    'product_id': p.product_id,
                    'base_currency': p.base_currency_id,
                    'quote_currency': p.quote_currency_id,
                    # Fallback to 0 if volume missing so sort works
                    'volume_24h': float(getattr(p, 'volume_24h', 0) or 0),
                })
        
        products.sort(key=lambda x: x['volume_24h'], reverse=True)
        return products

    def get_existing_range(self, product_id, granularity):
        result = self.db.execute("""
            SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
            FROM candles 
            WHERE product_id = ? AND granularity = ?
        """, [product_id, granularity]).fetchone()
        
        return result[0], result[1], result[2] or 0

    def fetch_candles(self, product_id, start, end, db_granularity="300"):
        candles = []
        current = start
        gran_seconds = int(db_granularity)
        max_seconds = MAX_CANDLES * gran_seconds
        
        # Mapping DuckDB granularity to Coinbase API format
        api_granularity = "FIVE_MINUTE" if db_granularity == "300" else "ONE_MINUTE"
        
        while current < end:
            chunk_end = min(current + max_seconds, end)
            try:
                resp = self.client.get_public_candles(
                    product_id=product_id,
                    start=str(int(current)),
                    end=str(int(chunk_end)),
                    granularity=api_granularity
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
                            'granularity': db_granularity,
                        })
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(5)
            
            current = chunk_end
            time.sleep(4)
        
        return candles

    def save_candles(self, candles):
        if not candles:
            return 0
        df = pd.DataFrame(candles)
        self.cache.save_candles(df)
        return len(candles)

    def ingest_product(self, product_id, days=365, granularity="300"):
        print(f"\n[{product_id}] ", end="", flush=True)
        
        min_ts, max_ts, count = self.get_existing_range(product_id, granularity)
        
        now = datetime.now()
        target_start = now - timedelta(days=days)
        
        if min_ts and max_ts:
            existing_range = (max_ts - min_ts).days
            target_range = (now - target_start).days
            if existing_range >= target_range * 0.95:
                print(f"Have {count} candles ({existing_range} days), skipping", flush=True)
                return count
                
        start_ts = int(min_ts.timestamp()) if min_ts else int(target_start.timestamp())
        end_ts = int(now.timestamp())
        
        candles = self.fetch_candles(product_id, start_ts, end_ts, granularity)
        
        if candles:
            saved = self.save_candles(candles)
            print(f"Saved {saved} candles", flush=True)
            return saved
            
        print("No data", flush=True)
        return 0

    def close(self):
        self.db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coinbase cross-pair candle ingestor")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--granularity", default="300", choices=["60", "300", "900", "3600", "21600", "86400"])
    parser.add_argument("--top", type=int, default=20, help="Top N pairs by volume")
    args = parser.parse_args()
    
    print("=== Fetching Cross Pairs for Dijkstra Map ===")
    ingestor = CrossPairIngestor()
    
    products = ingestor.get_cross_products()[:args.top]
    print(f"Found {len(products)} relevant cross pairs.")
    for p in products:
        print(f"  {p['product_id']}: ~${p['volume_24h']/1000:.1f}k")
        
    total = 0
    for i, p in enumerate(products):
        print(f"[{i+1}/{len(products)}] ", end="")
        try:
            total += ingestor.ingest_product(p['product_id'], args.days, args.granularity)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Failed: {e}")
            
    ingestor.close()
    print(f"\nSaved {total} total cross-pair candles.")
