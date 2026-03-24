import threading
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import duckdb
from coinbase.rest import RESTClient

from config import Config

CHUNK_CANDLES = 350
INITIAL_WORKERS = 8
MAX_RPS = 7.0

_db_lock = threading.Lock()


class _TokenBucket:
    def __init__(self, rate: float):
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
        time.sleep(wait)
        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0)


class CandleCache:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Config.DB_PATH)
        self.client = RESTClient(Config.COINBASE_API_KEY, Config.COINBASE_API_SECRET)
        self._init_db()
        self._concurrency = INITIAL_WORKERS
        self._concurrency_lock = threading.Lock()
        self._bucket = _TokenBucket(MAX_RPS)

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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_prod_gran ON candles(product_id, granularity)")

    def list_products(self, granularity: str = "300") -> List[str]:
        with duckdb.connect(self.db_path, read_only=True) as conn:
            results = conn.execute(
                "SELECT DISTINCT product_id FROM candles WHERE granularity = ?", [granularity]
            ).fetchall()
            return [r[0] for r in results]

    def query(self, sql: str, params: List = None) -> pd.DataFrame:
        with duckdb.connect(self.db_path, read_only=True) as conn:
            return conn.execute(sql, params or []).df()

    def save_candles(self, df: pd.DataFrame):
        if df.empty:
            return
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        with _db_lock:
            with duckdb.connect(self.db_path) as conn:
                conn.execute("INSERT OR IGNORE INTO candles SELECT * FROM df")

    def get_candles(self, product_id: str, start: datetime, end: datetime, granularity: str = "300", skip_fetch: bool = False) -> pd.DataFrame:
        with duckdb.connect(self.db_path, read_only=True) as conn:
            query = """
                SELECT * FROM candles
                WHERE product_id = ? AND granularity = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = conn.execute(query, [product_id, granularity, start, end]).df()

        if skip_fetch:
            return df

        gran_seconds = int(granularity)
        expected = int((end - start).total_seconds() / gran_seconds) + 1

        if len(df) < expected * 0.95:
            print(f"[{product_id}] miss ({len(df)}/{expected}), trying WS snapshot first...")
            try:
                self.ws_snapshot([product_id], granularity)
            except Exception as e:
                print(f"  [{product_id}] ws_snapshot failed: {e}")

            with duckdb.connect(self.db_path, read_only=True) as conn:
                df = conn.execute(query, [product_id, granularity, start, end]).df()

        return df

    def _fetch_chunk(self, product_id: str, chunk_start: datetime, chunk_end: datetime,
                     api_gran: str, granularity: str) -> int:
        try:
            self._bucket.acquire()
            resp = self.client.get_public_candles(
                product_id=product_id,
                start=str(int(chunk_start.timestamp())),
                end=str(int(chunk_end.timestamp())),
                granularity=api_gran,
            )
            if not (hasattr(resp, 'candles') and resp.candles):
                return 0
            rows = [{
                'product_id': product_id,
                'timestamp': datetime.fromtimestamp(int(c.start)),
                'open': float(c.open), 'high': float(c.high),
                'low': float(c.low),  'close': float(c.close),
                'volume': float(c.volume), 'granularity': granularity,
            } for c in resp.candles]
            df = pd.DataFrame(rows)
            self.save_candles(df)
            return len(rows)
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e):
                return -1
            print(f"  [{product_id}] chunk error: {e}")
            return 0

    def _fetch_and_save(self, product_id: str, start: datetime, end: datetime, granularity: str):
        api_gran = {
            "60": "ONE_MINUTE", "300": "FIVE_MINUTE", "900": "FIFTEEN_MINUTE",
            "3600": "ONE_HOUR", "21600": "SIX_HOUR", "86400": "ONE_DAY",
        }.get(granularity, "FIVE_MINUTE")

        gran_sec = int(granularity)
        chunk_sec = CHUNK_CANDLES * gran_sec

        chunks = []
        cur = start
        while cur < end:
            chunks.append((cur, min(cur + timedelta(seconds=chunk_sec), end)))
            cur += timedelta(seconds=chunk_sec)

        total = 0
        i = 0
        while i < len(chunks):
            result = self._fetch_chunk(product_id, chunks[i][0], chunks[i][1], api_gran, granularity)
            if result == -1:
                with self._concurrency_lock:
                    self._concurrency = max(1, self._concurrency - 2)
                    print(f"  429 hit — concurrency → {self._concurrency}, sleeping 15s")
                time.sleep(15)
            else:
                total += result
                i += 1

        print(f"  [{product_id}] {total} candles saved")

    def ws_snapshot(self, pairs: List[str], granularity: str = "300"):
        import json, threading
        from coinbase.websocket import WSClient

        received = set()
        done = threading.Event()
        rows_all = []
        lock = threading.Lock()

        def on_message(msg):
            try:
                data = json.loads(msg) if isinstance(msg, str) else msg
            except Exception:
                return
            for event in data.get("events", []):
                if event.get("type") != "snapshot":
                    continue
                for c in event.get("candles", []):
                    pid = c.get("product_id", "")
                    try:
                        rows_all.append({
                            'product_id': pid,
                            'timestamp': pd.to_datetime(int(c['start']), unit='s'),
                            'open': float(c['open']), 'high': float(c['high']),
                            'low': float(c['low']),   'close': float(c['close']),
                            'volume': float(c['volume']), 'granularity': granularity,
                        })
                        with lock:
                            received.add(pid)
                    except Exception:
                        pass
            with lock:
                if received.issuperset(set(pairs)):
                    done.set()

        client = WSClient(api_key=None, api_secret=None, on_message=on_message)
        client.open()
        client.subscribe(pairs, ["candles"])

        done.wait(timeout=60)
        client.close()

        if rows_all:
            df = pd.DataFrame(rows_all)
            self.save_candles(df)
            print(f"[WS snapshot] {len(rows_all)} candles from {len(received)}/{len(pairs)} pairs")

    def prefetch_all(self, pairs: List[str], start: datetime, end: datetime, granularity: str = "300"):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch_one(pid):
            self.get_candles(pid, start, end, granularity)
            return pid

        remaining = list(pairs)
        while remaining:
            with self._concurrency_lock:
                workers = self._concurrency
            batch = remaining[:workers * 2]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(fetch_one, pid): pid for pid in batch}
                done = []
                for f in as_completed(futures):
                    pid = futures[f]
                    try:
                        f.result()
                        done.append(pid)
                        print(f"  ✓ {pid} ({len(done)}/{len(batch)})")
                    except Exception as e:
                        print(f"  ✗ {pid}: {e}")
                        done.append(pid)
            remaining = [p for p in remaining if p not in done]