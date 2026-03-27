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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_params (
                    edge VARCHAR PRIMARY KEY,
                    curvature DOUBLE DEFAULT 2.0,
                    y_depth INTEGER DEFAULT 200,
                    x_pixels INTEGER DEFAULT 20,
                    fee_rate DOUBLE DEFAULT 0.001,
                    updated_at TIMESTAMP DEFAULT now()
                )
            """)

    def list_products(self, granularity: str = "300") -> List[str]:
        with duckdb.connect(self.db_path, read_only=True) as conn:
            results = conn.execute(
                "SELECT DISTINCT product_id FROM candles WHERE granularity = ?", [granularity]
            ).fetchall()
            return [r[0] for r in results]

    def cached_products_in_range(
        self,
        product_ids: List[str],
        start: datetime,
        end: datetime,
        granularity: str = "300",
    ) -> List[str]:
        if not product_ids:
            return []

        placeholders = ", ".join(["?"] * len(product_ids))
        sql = f"""
            SELECT DISTINCT product_id
            FROM candles
            WHERE granularity = ?
              AND product_id IN ({placeholders})
              AND timestamp BETWEEN ? AND ?
        """
        params = [granularity, *product_ids, start, end]
        with duckdb.connect(self.db_path, read_only=True) as conn:
            results = conn.execute(sql, params).fetchall()
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

    def get_candles(self, product_id: str, start: datetime, end: datetime, granularity: str = "300") -> pd.DataFrame:
        """Pure DuckDB read. All data ingestion must happen before this call."""
        with duckdb.connect(self.db_path, read_only=True) as conn:
            return conn.execute("""
                SELECT * FROM candles
                WHERE product_id = ? AND granularity = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, [product_id, granularity, start, end]).df()

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
            self._fetch_and_save(pid, start, end, granularity)
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

    def _binance_fetch_zip(self, url: str, dest: Path) -> bool:
        """
        GET url to dest if dest doesn't exist or is 0 bytes.
        Returns True on success, False on 404 or any error.
        File presence on disk IS the cache — no ETag, no HEAD.
        """
        import requests as req

        if dest.exists() and dest.stat().st_size > 0:
            return True

        try:
            resp = req.get(url, timeout=120)
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
        except Exception as e:
            print(f"  [Binance] {url}: {e}")
            return False

        return True

    def list_binance_pairs(self) -> List[str]:
        """
        Fetch all TRADING pairs from Binance exchangeInfo API.
        Returns raw product_ids in BASE-QUOTE format (e.g., BTC-USDT, ETH-BTC).
        Filters out fiat quotes and UP/DOWN/BULL/BEAR mutations only.
        """
        import requests as req
        
        FIAT_QUOTES = {'TRY', 'EUR', 'IDR', 'JPY', 'BRL', 'MXN', 'PLN', 'ARS', 'EURI', 'ZAR', 'UAH', 'COP', 'RUB', 'NGN'}
        
        resp = req.get("https://data-api.binance.vision/api/v3/exchangeInfo", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        pairs = []
        for sym in data.get("symbols", []):
            if sym.get("status") != "TRADING":
                continue
            base = sym.get("baseAsset", "")
            quote = sym.get("quoteAsset", "")
            if not base or not quote:
                continue
            
            if any(x in base or x in quote for x in ['UP', 'DOWN', 'BULL', 'BEAR']):
                continue
            
            if quote in FIAT_QUOTES:
                continue
            
            pairs.append(f"{base}-{quote}")
        
        return sorted(set(pairs))

    def import_binance_archive(self, pairs: List[str], granularity: str = "300") -> List[str]:
        """
        Draw-through: given a bag of pairs, fetch what is missing from disk for the
        3-year window and ingest into DuckDB resampled to granularity seconds.
        product_id stored in Coinbase style (USDT->USD).
        Cache dir: ~/mpdata/klines/1m/{BASE}/{QUOTE}/ (or $MP_IMPORT/klines/...).
        Returns product_ids successfully imported into DuckDB.
        """
        from datetime import date
        import zipfile

        mpdata = Path(os.environ.get("MP_IMPORT", Path.home() / "mpdata/import"))
        gran_sec = int(granularity)
        bucket_ms = gran_sec * 1000
        imported: List[str] = []

        today = date.today()
        cutoff_ts = pd.Timestamp.now() - pd.Timedelta(days=3 * 365)
        cutoff_ms = int(cutoff_ts.timestamp() * 1000)

        for product_id in pairs:
            base, quote = product_id.split("-", 1)
            binance_quote = "USDT" if quote == "USD" else quote
            if base == binance_quote:
                continue  # e.g. USDT-USDT — skip

            # Skip if DuckDB already has >1000 bars in the 3-year window
            with duckdb.connect(self.db_path, read_only=True) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM candles WHERE product_id=? AND granularity=? AND timestamp >= ?",
                    [product_id, granularity, cutoff_ts]
                ).fetchone()
            if row and row[0] > 1000:
                print(f"[Binance] {product_id}: already in DuckDB ({row[0]:,} bars), skipping")
                imported.append(product_id)
                continue

            symbol = f"{base}{binance_quote}"
            tf = "1m"
            cache_dir = mpdata / "klines" / "1m" / base / binance_quote
            cache_dir.mkdir(parents=True, exist_ok=True)

            zip_paths: List[Path] = []

            # Monthly zips: cutoff month up to (not including) current month
            yr, mo = today.year - 3, today.month
            while (yr, mo) < (today.year, today.month):
                fname = f"{symbol}-{tf}-{yr:04d}-{mo:02d}"
                zip_p = cache_dir / f"{fname}.zip"
                url = (f"https://data.binance.vision/data/spot/monthly/klines"
                       f"/{symbol}/{tf}/{fname}.zip")
                if self._binance_fetch_zip(url, zip_p):
                    zip_paths.append(zip_p)
                mo += 1
                if mo > 12:
                    mo = 1
                    yr += 1

            # Daily zips: days 1 through yesterday of the current partial month
            for day in range(1, today.day):
                d = date(today.year, today.month, day)
                fname = f"{symbol}-{tf}-{d.isoformat()}"
                zip_p = cache_dir / f"{fname}.zip"
                url = (f"https://data.binance.vision/data/spot/daily/klines"
                       f"/{symbol}/{tf}/{fname}.zip")
                if self._binance_fetch_zip(url, zip_p):
                    zip_paths.append(zip_p)

            if not zip_paths:
                print(f"[Binance] no archive data for {product_id}")
                continue

            # Extract CSVs from zips, read into DataFrames, then delete extracted CSVs
            try:
                dfs = []
                for zp in zip_paths:
                    try:
                        with zipfile.ZipFile(zp) as zf:
                            names = [n for n in zf.namelist() if n.endswith(".csv")]
                            if names:
                                dfs.append(pd.read_csv(zf.open(names[0]), header=None, usecols=[0, 1, 2, 3, 4, 5]))
                    except Exception:
                        pass
                if not dfs:
                    print(f"[Binance] {product_id}: could not read any CSVs")
                    continue

                df = pd.concat(dfs)
                df = df.drop_duplicates(subset=[0]).sort_values(0)
                df.columns = ["ts", "open", "high", "low", "close", "volume"]
                df = df.dropna()
                df["ts"] = df["ts"].astype("int64")
                df = df[df["ts"] >= cutoff_ms].reset_index(drop=True)
                if df.empty:
                    print(f"[Binance] {product_id}: no data in last 3 years")
                    continue

                df["bucket"] = (df["ts"] // bucket_ms) * bucket_ms
                resampled = df.groupby("bucket").agg(
                    open=("open",   "first"),
                    high=("high",   "max"),
                    low=("low",    "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                ).reset_index()
                resampled["timestamp"] = pd.to_datetime(resampled["bucket"], unit="ms")
                resampled["product_id"] = product_id
                resampled["granularity"] = granularity
                out = resampled[["product_id", "timestamp", "open", "high", "low", "close", "volume", "granularity"]]
                self.save_candles(out)
                yrs = len(out) * gran_sec / (365.25 * 86400)
                print(f"[Binance] {product_id}: {len(out):,} bars imported ({yrs:.1f}yr)")
                imported.append(product_id)
            except Exception as e:
                print(f"[Binance] {product_id}: {e}")

        return imported

    def save_edge_params(self, params: dict):
        """Upsert edge params. params = {edge_str: {curvature, y_depth, x_pixels, fee_rate}}"""
        with _db_lock:
            with duckdb.connect(self.db_path) as conn:
                for edge, p in params.items():
                    conn.execute("""
                        INSERT INTO edge_params (edge, curvature, y_depth, x_pixels, fee_rate, updated_at)
                        VALUES (?, ?, ?, ?, ?, now())
                        ON CONFLICT (edge) DO UPDATE SET
                            curvature=excluded.curvature,
                            y_depth=excluded.y_depth,
                            x_pixels=excluded.x_pixels,
                            fee_rate=excluded.fee_rate,
                            updated_at=excluded.updated_at
                    """, [edge, p['curvature'], p['y_depth'], p['x_pixels'], p['fee_rate']])

    def load_edge_params(self) -> dict:
        """Returns {edge_str: {curvature, y_depth, x_pixels, fee_rate}}"""
        with duckdb.connect(self.db_path, read_only=True) as conn:
            rows = conn.execute(
                "SELECT edge, curvature, y_depth, x_pixels, fee_rate FROM edge_params"
            ).fetchall()
        return {r[0]: {'curvature': r[1], 'y_depth': r[2], 'x_pixels': r[3], 'fee_rate': r[4]} for r in rows}
