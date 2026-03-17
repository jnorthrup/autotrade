import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import duckdb
from coinbase.rest import RESTClient

from config import Config

class CandleCache:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Config.DB_PATH)
        self.client = RESTClient(Config.COINBASE_API_KEY, Config.COINBASE_API_SECRET)
        self._init_db()

    def _init_db(self):
        with duckdb.connect(self.db_path) as conn:
            conn.execute("""
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_product_gran ON candles(product_id, granularity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON candles(timestamp)")

    def list_products(self, granularity: str = "300") -> List[str]:
        """Returns a list of all product IDs present for a given granularity."""
        with duckdb.connect(self.db_path, read_only=True) as conn:
            query = "SELECT DISTINCT product_id FROM candles WHERE granularity = ?"
            results = conn.execute(query, [granularity]).fetchall()
            return [r[0] for r in results]

    def query(self, sql: str, params: List = None) -> pd.DataFrame:
        """Generic read-only query helper."""
        with duckdb.connect(self.db_path, read_only=True) as conn:
            return conn.execute(sql, params or []).df()

    def get_candles(self, product_id: str, start: datetime, end: datetime, granularity: str = "300") -> pd.DataFrame:
        """
        Main draw-through entry point. 
        Returns candles for the range, fetching from API if local data is missing.
        """
        # 1. Check local coverage
        with duckdb.connect(self.db_path, read_only=True) as conn:
            query = """
                SELECT * FROM candles 
                WHERE product_id = ? AND granularity = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = conn.execute(query, [product_id, granularity, start, end]).df()

        # 2. Heuristic check: is the data complete enough?
        # Simple check: calculate expected number of candles
        gran_seconds = int(granularity)
        expected_candles = int((end - start).total_seconds() / gran_seconds) + 1
        
        # If we have less than 95% of expected candles, or no data at all, fetch from API
        if len(df) < expected_candles * 0.95:
            print(f"[{product_id}] Cache miss or gap detected ({len(df)}/{expected_candles}). Drawing through Coinbase API...")
            self._fetch_and_save(product_id, start, end, granularity)
            
            # Re-query after fetch
            with duckdb.connect(self.db_path, read_only=True) as conn:
                df = conn.execute(query, [product_id, granularity, start, end]).df()
        
        return df

    def _fetch_and_save(self, product_id: str, start: datetime, end: datetime, granularity: str):
        """Fetches from Coinbase and performs idempotent write to DuckDB."""
        
        api_granularity = {
            "60": "ONE_MINUTE",
            "300": "FIVE_MINUTE",
            "900": "FIFTEEN_MINUTE",
            "3600": "ONE_HOUR",
            "21600": "SIX_HOUR",
            "86400": "ONE_DAY"
        }.get(granularity, "FIVE_MINUTE")
        
        # Coinbase limit is ~300 candles per request
        current = start
        gran_seconds = int(granularity)
        max_seconds = 300 * gran_seconds
        
        all_candles = []
        
        while current < end:
            chunk_end = min(current + timedelta(seconds=max_seconds), end)
            
            try:
                resp = self.client.get_public_candles(
                    product_id=product_id,
                    start=str(int(current.timestamp())),
                    end=str(int(chunk_end.timestamp())),
                    granularity=api_granularity
                )
                
                if hasattr(resp, 'candles') and resp.candles:
                    chunk_candles = []
                    for c in resp.candles:
                        chunk_candles.append({
                            'product_id': product_id,
                            'timestamp': datetime.fromtimestamp(int(c.start)),
                            'open': float(c.open),
                            'high': float(c.high),
                            'low': float(c.low),
                            'close': float(c.close),
                            'volume': float(c.volume),
                            'granularity': granularity,
                        })
                    
                    if chunk_candles:
                        df = pd.DataFrame(chunk_candles)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        with duckdb.connect(self.db_path) as conn:
                            conn.execute("INSERT OR REPLACE INTO candles SELECT * FROM df")
                        print(f"  Saved {len(chunk_candles)} candles to DuckDB...")
                
            except Exception as e:
                # Tier 3 Persistent intelligence: handle API backoff
                if "429" in str(e) or "Too Many Requests" in str(e):
                    print("  Rate limited. Backing off 10s...")
                    time.sleep(10)
                else:
                    print(f"  API Error for {product_id}: {e}")
                    time.sleep(2)
            
            current = chunk_end
            time.sleep(1.0) # Respect rate limits

if __name__ == "__main__":
    # Self-test
    cache = CandleCache()
    now = datetime.now()
    start = now - timedelta(hours=2)
    test_df = cache.get_candles("BTC-USD", start, now)
    print(f"Retrieved {len(test_df)} candles for BTC-USD.")
