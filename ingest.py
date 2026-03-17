#!/usr/bin/env python3
"""
Production candle ingestion system for Coinbase Advanced.
Uses DuckDB for efficient storage + async fetching.

Usage:
    python3 ingest.py                    # Run once
    python3 ingest.py --watch           # Continuous mode
    python3 ingest.py --pairs BTC-USD   # Specific pairs
"""

import os
import sys
import time
import signal
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import duckdb
import requests

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from candle_cache import CandleCache

DB_PATH = Config.DB_PATH
CACHE_DIR = Path(__file__).parent / "candle_cache_1m"

MAX_CANDLES = 350  # Coinbase API limit

class CandleIngestor:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key or Config.COINBASE_API_KEY
        self.api_secret = api_secret or Config.COINBASE_API_SECRET
        self.client = RESTClient(self.api_key, self.api_secret)
        self.cache = CandleCache(str(DB_PATH))
        self.db = duckdb.connect(str(DB_PATH))
        self._init_db()
        self.running = True
    
    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                product_id VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                granularity VARCHAR,
                PRIMARY KEY (product_id, timestamp, granularity)
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_product ON candles(product_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_time ON candles(timestamp)")
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id VARCHAR PRIMARY KEY,
                base_currency VARCHAR,
                quote_currency VARCHAR,
                status VARCHAR,
                volume_24h DOUBLE
            )
        """)
    
    def get_products(self, min_volume=100_000):
        resp = self.client.get_public_products()
        
        # Stablecoins have 0 momentum - exclude from quote side
        STABLECOINS = {'USD', 'USDC', 'USDT', 'DAI', 'FRAX', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'USDP'}
        
        products = []
        
        for p in resp.products:
            if p.status != "online" or p.trading_disabled:
                continue
            
            vol = float(getattr(p, 'approximate_quote_24h_volume', 0) or 0)
            if vol < min_volume:
                continue
            
            quote = p.quote_currency_id.upper()
            
            # Skip if quote is stablecoin (0 momentum)
            # But include everything else - including BTC-ETH, BTC-BTC, etc
            if quote in STABLECOINS:
                continue
            
            products.append({
                'product_id': p.product_id,
                'base_currency': p.base_currency_id,
                'quote_currency': p.quote_currency_id,
                'status': p.status,
                'volume_24h': vol,
            })
        
        products.sort(key=lambda x: x['volume_24h'], reverse=True)
        return products
    
    def save_products(self, products):
        if not products:
            return
        df = pd.DataFrame(products)
        df['last_updated'] = datetime.now()
        
        # Clear and insert
        self.db.execute("DELETE FROM products")
        for _, row in df.iterrows():
            self.db.execute("""
                INSERT INTO products VALUES (?, ?, ?, ?, ?)
            """, [row['product_id'], row['base_currency'], row['quote_currency'], 
                  row['status'], row['volume_24h']])
    
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
        
        api_granularity = {
            "60": "ONE_MINUTE",
            "300": "FIVE_MINUTE",
            "900": "FIFTEEN_MINUTE",
            "3600": "ONE_HOUR",
            "21600": "SIX_HOUR",
            "86400": "ONE_DAY"
        }.get(db_granularity, "FIVE_MINUTE")
        
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
            time.sleep(4)  # Rate limit
        
        return candles
    
    def save_candles(self, candles):
        if not candles:
            return 0
        
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        self.db.execute("""
            INSERT OR REPLACE INTO candles 
            SELECT * FROM df
        """)
        
        return len(candles)
    
    def ingest_product(self, product_id, days=365, granularity="300"):
        print(f"\n[{product_id}] ", end="", flush=True)
        start = datetime.now() - timedelta(days=days)
        end = datetime.now()
        
        # Use the draw-through cache
        df = self.cache.get_candles(product_id, start, end, granularity)
        return len(df)
    
    def run(self, products, days=365, granularity="ONE_MINUTE"):
        total = 0
        for i, p in enumerate(products):
            if not self.running:
                break
            
            print(f"[{i+1}/{len(products)}] ", end="", flush=True)
            
            try:
                count = self.ingest_product(p['product_id'], days, granularity)
                total += count
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"Failed: {e}")
        
        return total
    
    def query(self, sql):
        return self.db.execute(sql).df()
    
    def close(self):
        self.db.close()

def signal_handler(signum, frame):
    print("\n\nShutting down...")
    ingestor.running = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coinbase candle ingestor")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--granularity", default="300", 
                       choices=["60", "300", "900", "3600", "21600", "86400"])
    parser.add_argument("--top", type=int, default=100, help="Top N pairs by volume")
    parser.add_argument("--pairs", nargs="+", help="Specific pairs")
    parser.add_argument("--min-volume", type=float, default=100_000, help="Min 24h volume")
    args = parser.parse_args()
    
    print(f"=== Coinbase Candle Ingestor ===")
    print(f"Database: {DB_PATH}")
    print(f"Days: {args.days}, Granularity: {args.granularity}")
    print()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    ingestor = CandleIngestor()
    
    # Get products
    if args.pairs:
        products = [{'product_id': p} for p in args.pairs]
    else:
        print("Fetching product list...")
        products = ingestor.get_products(min_volume=args.min_volume)[:args.top]
        ingestor.save_products(products)
    
    print(f"Products: {len(products)}")
    for i, p in enumerate(products[:10]):
        print(f"  {i+1}. {p['product_id']}: ${p.get('volume_24h', 0)/1e6:.1f}M")
    print(f"  ... and {len(products)-10} more")
    print()
    
    # Run ingestion
    total = ingestor.run(products, args.days, args.granularity)
    
    print(f"\n=== Done: {total:,} total candles ===")
    
    # Show stats
    stats = ingestor.query("""
        SELECT product_id, COUNT(*) as n_candles, 
               MIN(timestamp) as first_candle,
               MAX(timestamp) as last_candle
        FROM candles 
        WHERE granularity = 'ONE_MINUTE'
        GROUP BY product_id 
        ORDER BY n_candles DESC 
        LIMIT 10
    """)
    print("\nTop pairs by candle count:")
    print(stats.to_string(index=False))
    
    ingestor.close()
