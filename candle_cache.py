import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import duckdb
from coinbase.rest import RESTClient

from config import Config
from pool_client import PoolClient, pool_is_running, DEFAULT_SOCKET

CHUNK_CANDLES = 350
INITIAL_WORKERS = 8
MAX_RPS = 7.0

_db_lock = threading.Lock()

def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _max_reasonable_future() -> pd.Timestamp:
    return pd.Timestamp(_utc_now_naive()) + pd.Timedelta(days=1)


def _expected_candle_count(start: datetime, end: datetime, granularity: str) -> int:
    _, _, count = _expected_bucket_bounds(start, end, granularity)
    return count


def _expected_bucket_bounds(
    start: datetime,
    end: datetime,
    granularity: str,
) -> Tuple[Optional[int], Optional[int], int]:
    if end <= start:
        return None, None, 0
    gran_ns = _granularity_seconds(granularity) * 1_000_000_000
    start_ns = _timestamp_ns(start)
    end_ns = _timestamp_ns(end)
    first_ns = ((start_ns + gran_ns - 1) // gran_ns) * gran_ns
    if first_ns >= end_ns:
        return None, None, 0
    last_ns = ((end_ns - 1) // gran_ns) * gran_ns
    count = int(((last_ns - first_ns) // gran_ns) + 1)
    return first_ns, last_ns, count


def _use_pool(db_path: Optional[str] = None) -> bool:
    """True if literbike pool server is running and the canonical DB is in use."""
    try:
        if not pool_is_running():
            return False
        if db_path is None:
            return True
        resolved = Path(db_path).resolve()
        return resolved == Path(Config.DB_PATH).resolve() and resolved.exists()
    except Exception:
        return False


def _pool() -> PoolClient:
    """Get a PoolClient connected to the shared server."""
    return PoolClient(DEFAULT_SOCKET)


def _sql_escape(val) -> str:
    """Convert a Python value to a SQL literal."""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int,)):
        return str(val)
    if isinstance(val, (float,)):
        return repr(val)
    if isinstance(val, (pd.Timestamp, datetime)):
        return f"'{val.isoformat()}'"
    if isinstance(val, str):
        escaped = val.replace("'", "''")
        return f"'{escaped}'"
    return f"'{val}'"


def _parse_candle_timestamp(raw_value) -> pd.Timestamp:
    if raw_value is None:
        raise ValueError("Missing candle timestamp")
    if isinstance(raw_value, pd.Timestamp):
        ts = raw_value
    elif isinstance(raw_value, datetime):
        ts = pd.Timestamp(raw_value)
    elif isinstance(raw_value, str):
        value = raw_value.strip()
        if value.isdigit():
            return _parse_candle_timestamp(int(value))
        ts = pd.Timestamp(value)
    else:
        value = int(raw_value)
        abs_value = abs(value)
        if abs_value >= 10**18:
            ts = pd.to_datetime(value, unit="ns")
        elif abs_value >= 10**15:
            ts = pd.to_datetime(value, unit="us")
        elif abs_value >= 10**12:
            ts = pd.to_datetime(value, unit="ms")
        else:
            ts = pd.to_datetime(value, unit="s")

    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return pd.Timestamp(ts)


def _granularity_seconds(granularity: str) -> int:
    return max(1, int(granularity))


def _timestamp_ns(value: datetime) -> int:
    return int(pd.Timestamp(value).value)


def _epoch_seconds_utc(value: datetime) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def _floor_time_to_granularity(value: datetime, granularity: str) -> datetime:
    gran_ns = _granularity_seconds(granularity) * 1_000_000_000
    ts_ns = _timestamp_ns(value)
    return pd.Timestamp((ts_ns // gran_ns) * gran_ns).to_pydatetime()


def _ceil_time_to_granularity(value: datetime, granularity: str) -> datetime:
    gran_ns = _granularity_seconds(granularity) * 1_000_000_000
    ts_ns = _timestamp_ns(value)
    if ts_ns % gran_ns == 0:
        return pd.Timestamp(ts_ns).to_pydatetime()
    return pd.Timestamp(((ts_ns // gran_ns) + 1) * gran_ns).to_pydatetime()


def _dedupe_subscriptions(subscriptions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: Dict[Tuple[str, str], Dict[str, str]] = {}
    for sub in subscriptions or []:
        exchange = str(sub.get("exchange") or "").strip()
        product_id = str(sub.get("product_id") or "").strip()
        if not exchange or not product_id or "-" not in product_id:
            continue
        deduped[(exchange, product_id)] = {"exchange": exchange, "product_id": product_id}
    return list(deduped.values())


def _bag_surface_name(
    subscriptions: List[Dict[str, str]],
    start: datetime,
    end: datetime,
    granularity: str,
) -> str:
    return f"bag_surface_{_bag_digest(subscriptions, start, end, granularity)}"


def _bag_digest(
    subscriptions: List[Dict[str, str]],
    start: datetime,
    end: datetime,
    granularity: str,
) -> str:
    keys = [f"{sub['exchange']}:{sub['product_id']}" for sub in _dedupe_subscriptions(subscriptions)]
    return hashlib.sha1(
        json.dumps(
            {
                "subscriptions": sorted(keys),
                "start": pd.Timestamp(start).isoformat(),
                "end": pd.Timestamp(end).isoformat(),
                "granularity": str(granularity),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def _bag_definition_id(subscriptions: List[Dict[str, str]]) -> str:
    keys = [f"{sub['exchange']}:{sub['product_id']}" for sub in _dedupe_subscriptions(subscriptions)]
    return hashlib.sha1(
        json.dumps({"subscriptions": sorted(keys)}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _bag_window_id(
    subscriptions: List[Dict[str, str]],
    start: datetime,
    end: datetime,
    granularity: str,
    min_coverage_ratio: float = 0.0,
) -> str:
    return hashlib.sha1(
        json.dumps(
            {
                "bag_id": _bag_definition_id(subscriptions),
                "start": pd.Timestamp(start).isoformat(),
                "end": pd.Timestamp(end).isoformat(),
                "granularity": str(granularity),
                "min_coverage_ratio": float(min_coverage_ratio),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def _bag_surface_view_name_for_window(bag_window_id: str) -> str:
    return f"bag_surface_{bag_window_id}"


def _bag_thresholds_view_name(
    subscriptions: List[Dict[str, str]],
    start: datetime,
    end: datetime,
    granularity: str,
    min_coverage_ratio: float = 0.0,
) -> str:
    return f"bag_thresholds_{_bag_window_id(subscriptions, start, end, granularity, min_coverage_ratio)}"


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
        self._ensure_candles_schema()
        stmts = [
            "CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_candles_exchange_time ON candles(exchange, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_candles_exchange_prod_gran ON candles(exchange, product_id, granularity)",
            """CREATE TABLE IF NOT EXISTS bags (
                bag_id VARCHAR PRIMARY KEY,
                bag_name VARCHAR,
                created_at TIMESTAMP DEFAULT now()
            )""",
            """CREATE TABLE IF NOT EXISTS bag_members (
                bag_id VARCHAR,
                exchange VARCHAR,
                product_id VARCHAR,
                PRIMARY KEY (bag_id, exchange, product_id)
            )""",
            """CREATE TABLE IF NOT EXISTS bag_windows (
                bag_window_id VARCHAR PRIMARY KEY,
                bag_id VARCHAR,
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                granularity VARCHAR,
                min_coverage_ratio DOUBLE,
                created_at TIMESTAMP DEFAULT now()
            )""",
            """CREATE TABLE IF NOT EXISTS bag_member_status (
                bag_window_id VARCHAR,
                exchange VARCHAR,
                product_id VARCHAR,
                covered BOOLEAN,
                expected_count BIGINT,
                actual_count BIGINT,
                coverage_ratio DOUBLE,
                min_coverage_ratio DOUBLE,
                meets_threshold BOOLEAN,
                bad_gap_count BIGINT,
                first_timestamp TIMESTAMP,
                last_timestamp TIMESTAMP,
                PRIMARY KEY (bag_window_id, exchange, product_id)
            )""",
            """CREATE TABLE IF NOT EXISTS edge_params (
                edge VARCHAR PRIMARY KEY,
                curvature DOUBLE DEFAULT 2.0,
                y_depth INTEGER DEFAULT 200,
                x_pixels INTEGER DEFAULT 20,
                fee_rate DOUBLE DEFAULT 0.001,
                updated_at TIMESTAMP DEFAULT now()
            )""",
            """CREATE OR REPLACE VIEW bag_thresholds_v AS
                SELECT
                    bw.bag_window_id,
                    bw.bag_id,
                    b.bag_name,
                    bw.window_start,
                    bw.window_end,
                    bw.granularity,
                    bw.min_coverage_ratio,
                    bms.exchange,
                    bms.product_id,
                    bms.covered,
                    bms.expected_count,
                    bms.actual_count,
                    bms.coverage_ratio,
                    bms.meets_threshold,
                    bms.bad_gap_count,
                    bms.first_timestamp,
                    bms.last_timestamp
                FROM bag_windows AS bw
                JOIN bag_member_status AS bms
                  ON bms.bag_window_id = bw.bag_window_id
                LEFT JOIN bags AS b
                  ON b.bag_id = bw.bag_id
            """,
        ]
        if _use_pool(self.db_path):
            p = _pool()
            for s in stmts:
                p.execute(s)
        else:
            with duckdb.connect(self.db_path) as conn:
                for s in stmts:
                    conn.execute(s)

    def _create_canonical_candles_table(self, executor) -> None:
        executor(
            """CREATE TABLE IF NOT EXISTS candles (
                exchange VARCHAR,
                product_id VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                granularity VARCHAR,
                PRIMARY KEY (exchange, product_id, timestamp, granularity)
            )"""
        )

    def _candles_table_info(self) -> List[List[object]]:
        sql = "PRAGMA table_info('candles')"
        try:
            if _use_pool(self.db_path):
                return list(_pool().execute(sql))
            with duckdb.connect(self.db_path) as conn:
                return conn.execute(sql).fetchall()
        except Exception:
            return []

    def _candles_schema_is_canonical(self, table_info: List[List[object]]) -> bool:
        if not table_info:
            return False
        names = {str(row[1]) for row in table_info}
        required = {
            "exchange",
            "product_id",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "granularity",
        }
        if not required.issubset(names):
            return False
        pk_columns = {str(row[1]) for row in table_info if bool(row[5])}
        expected_pk = {"exchange", "product_id", "timestamp", "granularity"}
        return expected_pk.issubset(pk_columns)

    def _migrate_legacy_candles_schema(self, executor, table_info: List[List[object]]) -> None:
        column_names = {str(row[1]) for row in table_info}
        if "product_id" not in column_names or "timestamp" not in column_names or "granularity" not in column_names:
            raise RuntimeError("Legacy candles schema is missing core columns and cannot be migrated safely")
        exchange_expr = "COALESCE(exchange, 'coinbase')" if "exchange" in column_names else "'coinbase'"
        print("[CandleCache] migrating legacy candles schema to exchange-qualified canonical table")
        try:
            executor("BEGIN TRANSACTION")
            executor("DROP TABLE IF EXISTS candles__exchange_keyed_migration")
            executor(
                """CREATE TABLE candles__exchange_keyed_migration (
                    exchange VARCHAR,
                    product_id VARCHAR,
                    timestamp TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    granularity VARCHAR,
                    PRIMARY KEY (exchange, product_id, timestamp, granularity)
                )"""
            )
            executor(
                f"""INSERT INTO candles__exchange_keyed_migration (
                        exchange,
                        product_id,
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        granularity
                    )
                    SELECT
                        {exchange_expr} AS exchange,
                        product_id,
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        granularity
                    FROM candles"""
            )
            executor("DROP TABLE candles")
            executor("ALTER TABLE candles__exchange_keyed_migration RENAME TO candles")
            executor("COMMIT")
        except Exception:
            try:
                executor("ROLLBACK")
            except Exception:
                pass
            raise

    def _ensure_candles_schema(self) -> None:
        table_info = self._candles_table_info()
        if _use_pool(self.db_path):
            executor = _pool().execute
            if not table_info:
                self._create_canonical_candles_table(executor)
                return
            if not self._candles_schema_is_canonical(table_info):
                raise RuntimeError(
                    "Legacy candles schema detected behind duckdb_pool. "
                    "Stop the pool and rerun once so CandleCache can migrate the file directly."
                )
            return

        with duckdb.connect(self.db_path) as conn:
            executor = conn.execute
            if not table_info:
                self._create_canonical_candles_table(executor)
                return
            if not self._candles_schema_is_canonical(table_info):
                self._migrate_legacy_candles_schema(executor, table_info)

    def bootstrap_database(
        self,
        granularity: Optional[str] = None,
        exchange: Optional[str] = None,
        future_days: int = 1,
    ) -> Dict[str, object]:
        self._ensure_candles_schema()
        normalized = self.normalize_candle_timestamps(
            granularity=granularity,
            exchange=exchange,
        )
        purged = self.purge_future_candles(
            granularity=granularity,
            future_days=future_days,
            exchange=exchange,
        )
        return {
            "db_path": str(Path(self.db_path).resolve()),
            "granularity": granularity,
            "exchange": exchange,
            "normalized_timestamps": int(normalized),
            "purged_future_rows": int(purged),
            "pool_enabled": bool(_use_pool(self.db_path)),
        }

    def list_products(self, granularity: str = "300", exchange: Optional[str] = None) -> List[str]:
        where = ["granularity = ?"]
        params: List = [granularity]
        if exchange is not None:
            where.append("exchange = ?")
            params.append(exchange)
        sql = f"SELECT DISTINCT product_id FROM candles WHERE {' AND '.join(where)}"
        if _use_pool(self.db_path):
            rows = _pool().execute(
                sql,
                params,
            )
            return [r[0] for r in rows]
        with duckdb.connect(self.db_path) as conn:
            results = conn.execute(sql, params).fetchall()
            return [r[0] for r in results]

    def historical_products(
        self,
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: Optional[str] = None,
        min_coverage_ratio: float = 0.9,
    ) -> List[str]:
        expected = _expected_candle_count(start, end, granularity)
        if expected <= 0:
            return []

        where = ["granularity = ?"]
        params: List = [granularity]
        if exchange is not None:
            where.append("exchange = ?")
            params.append(exchange)
        where.append("timestamp >= ? AND timestamp < ?")
        params.extend([start, end])
        sql = f"""
            SELECT product_id, COUNT(*) AS n_rows
            FROM candles
            WHERE {' AND '.join(where)}
            GROUP BY product_id
        """
        if _use_pool(self.db_path):
            rows = _pool().execute(sql, params)
        else:
            with duckdb.connect(self.db_path) as conn:
                rows = conn.execute(sql, params).fetchall()

        threshold = max(1, int(expected * max(0.0, min_coverage_ratio)))
        products = [product_id for product_id, n_rows in rows if int(n_rows) >= threshold]
        return sorted(set(products))

    def cached_products_in_range(
        self,
        product_ids: List[str],
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: Optional[str] = None,
    ) -> List[str]:
        if not product_ids:
            return []
        if exchange is None:
            statuses = [
                self.range_coverage_status(pid, start, end, granularity, exchange=None)
                for pid in product_ids
            ]
        else:
            statuses = self.verify_bag_contiguous_coverage(
                [{"exchange": exchange, "product_id": pid} for pid in product_ids],
                start,
                end,
                granularity=granularity,
            )
        return [str(status["product_id"]) for status in statuses if bool(status["covered"])]

    def query(self, sql: str, params: List = None) -> pd.DataFrame:
        if _use_pool(self.db_path):
            return _pool().execute_df(sql, params)
        with duckdb.connect(self.db_path) as conn:
            return conn.execute(sql, params or []).df()

    def _sanitize_candle_frame(
        self,
        df: pd.DataFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        clean = df.copy()
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
        clean = clean.dropna(subset=["timestamp"])
        clean = clean[clean["timestamp"] <= _max_reasonable_future()]
        if start is not None:
            clean = clean[clean["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            clean = clean[clean["timestamp"] < pd.Timestamp(end)]
        clean = clean.drop_duplicates(subset=["timestamp"], keep="last")
        clean = clean.sort_values("timestamp").reset_index(drop=True)
        return clean

    def candle_coverage_ratio(
        self,
        df: pd.DataFrame,
        start: datetime,
        end: datetime,
        granularity: str = "300",
    ) -> float:
        expected = _expected_candle_count(start, end, granularity)
        if expected <= 0:
            return 1.0
        return min(1.0, len(df) / expected)

    def get_candles_verified(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: Optional[str] = None,
        min_coverage_ratio: float = 0.0,
    ) -> Tuple[pd.DataFrame, float]:
        status = self.range_coverage_status(
            product_id,
            start,
            end,
            granularity,
            exchange=exchange,
        )
        coverage = float(status["coverage_ratio"])
        df = self.get_candles(product_id, start, end, granularity, exchange=exchange)
        if not bool(status["covered"]) or coverage < max(0.0, min_coverage_ratio):
            return pd.DataFrame(columns=df.columns), coverage
        return df, coverage

    def save_candles(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df[df['timestamp'] <= _max_reasonable_future()]
        if 'exchange' not in df.columns:
            raise ValueError("candles frame must include an exchange column")
        df = df.drop_duplicates(subset=['exchange', 'product_id', 'timestamp', 'granularity'], keep='last')
        if df.empty:
            return
        if _use_pool(self.db_path):
            self._save_candles_via_pool(df)
        else:
            with _db_lock:
                with duckdb.connect(self.db_path) as conn:
                    conn.register("df", df)
                    conn.execute("INSERT OR REPLACE INTO candles SELECT * FROM df")

    def _save_candles_via_pool(self, df: pd.DataFrame):
        """Upsert candles through the pool server using batch INSERT."""
        p = _pool()
        cols = [
            "exchange",
            "product_id",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "granularity",
        ]
        batch_size = 500
        for i in range(0, len(df), batch_size):
            chunk = df.iloc[i:i+batch_size]
            values_parts = []
            for _, row in chunk.iterrows():
                vals = ", ".join(_sql_escape(row[c]) for c in cols)
                values_parts.append(f"({vals})")
            sql = f"INSERT OR REPLACE INTO candles ({', '.join(cols)}) VALUES {', '.join(values_parts)}"
            p.execute(sql)

    def latest_timestamp(self, product_id: str, granularity: str = "300", exchange: Optional[str] = None) -> Optional[datetime]:
        where = ["product_id = ?", "granularity = ?"]
        params: List = [product_id, granularity]
        if exchange is not None:
            where.insert(0, "exchange = ?")
            params.insert(0, exchange)
        sql = f"""
            SELECT epoch_ns(MAX(timestamp))
            FROM candles
            WHERE {' AND '.join(where)}
              AND timestamp <= now() + INTERVAL 1 DAY
        """
        if _use_pool(self.db_path):
            rows = _pool().execute(sql, params)
            value = rows[0][0] if rows and rows[0] else None
        else:
            with duckdb.connect(self.db_path) as conn:
                row = conn.execute(sql, params).fetchone()
                value = row[0] if row else None
        if value is None:
            return None
        return _parse_candle_timestamp(value).to_pydatetime()

    def purge_future_candles(
        self,
        granularity: Optional[str] = None,
        future_days: int = 1,
        exchange: Optional[str] = None,
    ) -> int:
        cutoff = _utc_now_naive() + timedelta(days=future_days)
        where = "timestamp > ?"
        params = [cutoff]
        if granularity is not None:
            where += " AND granularity = ?"
            params.append(granularity)
        if exchange is not None:
            where += " AND exchange = ?"
            params.append(exchange)

        count_sql = f"SELECT COUNT(*) FROM candles WHERE {where}"
        delete_sql = f"DELETE FROM candles WHERE {where}"
        if _use_pool(self.db_path):
            rows = _pool().execute(count_sql, params)
            count = int(rows[0][0]) if rows and rows[0] else 0
            if count:
                _pool().execute(delete_sql, params)
            return count
        with _db_lock:
            with duckdb.connect(self.db_path) as conn:
                row = conn.execute(count_sql, params).fetchone()
                count = int(row[0]) if row else 0
                if count:
                    conn.execute(delete_sql, params)
        return count

    def normalize_candle_timestamps(
        self,
        granularity: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> int:
        filters: List[str] = []
        params: List[object] = []
        if granularity is not None:
            filters.append("granularity = ?")
            params.append(granularity)
        if exchange is not None:
            filters.append("exchange = ?")
            params.append(exchange)
        filters.append(
            "epoch_ns(timestamp) % (CAST(granularity AS BIGINT) * 1000000000) != 0"
        )
        where_sql = " AND ".join(filters)
        count_sql = f"SELECT COUNT(*) FROM candles WHERE {where_sql}"
        temp_sql = f"""
            CREATE TEMP TABLE candles_ts_normalized AS
            SELECT
                exchange,
                product_id,
                CAST(
                    to_timestamp(
                        FLOOR(epoch_ns(timestamp) / (CAST(granularity AS BIGINT) * 1000000000))
                        * CAST(granularity AS BIGINT)
                    ) AS TIMESTAMP
                ) AS timestamp,
                open,
                high,
                low,
                close,
                volume,
                granularity
            FROM candles
            WHERE {where_sql}
        """
        delete_sql = f"DELETE FROM candles WHERE {where_sql}"

        if _use_pool(self.db_path):
            raise RuntimeError("Timestamp normalization must run on direct DuckDB, not duckdb_pool")

        with _db_lock:
            with duckdb.connect(self.db_path) as conn:
                row = conn.execute(count_sql, params).fetchone()
                affected = int(row[0]) if row else 0
                if affected <= 0:
                    return 0
                conn.execute("DROP TABLE IF EXISTS candles_ts_normalized")
                conn.execute(temp_sql, params)
                conn.execute(delete_sql, params)
                conn.execute(
                    "INSERT OR REPLACE INTO candles SELECT * FROM candles_ts_normalized"
                )
                conn.execute("DROP TABLE candles_ts_normalized")
        return affected

    def get_candles(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: Optional[str] = None,
    ) -> pd.DataFrame:
        """Pure DuckDB read. All data ingestion must happen before this call."""
        where = ["product_id = ?", "granularity = ?", "timestamp >= ?", "timestamp < ?"]
        params: List = [product_id, granularity, start, end]
        if exchange is not None:
            where.insert(0, "exchange = ?")
            params.insert(0, exchange)
        sql = f"""
            SELECT * FROM candles
            WHERE {' AND '.join(where)}
            ORDER BY timestamp
        """
        if _use_pool(self.db_path):
            rows = _pool().execute(sql, params)
            if not rows:
                return pd.DataFrame()
            return self._sanitize_candle_frame(
                pd.DataFrame(
                    rows,
                    columns=[
                        "exchange",
                        "product_id",
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "granularity",
                    ],
                ),
                start=start,
                end=end,
            )
        with duckdb.connect(self.db_path) as conn:
            df = conn.execute(sql, params).df()
        return self._sanitize_candle_frame(df, start=start, end=end)

    def range_coverage_status(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: Optional[str] = None,
    ) -> Dict[str, object]:
        gran_sec = _granularity_seconds(granularity)
        gran_ns = gran_sec * 1_000_000_000
        expected_first_ns, expected_last_ns, expected = _expected_bucket_bounds(start, end, granularity)
        subscription_key = f"{exchange}:{product_id}" if exchange else product_id
        if expected <= 0:
            return {
                "exchange": exchange,
                "product_id": product_id,
                "subscription_key": subscription_key,
                "covered": True,
                "expected_count": 0,
                "actual_count": 0,
                "coverage_ratio": 1.0,
                "bad_gap_count": 0,
                "first_timestamp": None,
                "last_timestamp": None,
            }

        where = ["product_id = ?", "granularity = ?", "timestamp >= ?", "timestamp < ?"]
        params: List = [product_id, granularity, start, end, gran_ns]
        if exchange is not None:
            where.insert(0, "exchange = ?")
            params.insert(0, exchange)
        sql = f"""
            WITH ordered AS (
                SELECT
                    epoch_ns(timestamp) AS ts_ns,
                    LAG(epoch_ns(timestamp)) OVER (ORDER BY timestamp) AS prev_ts_ns
                FROM candles
                WHERE {' AND '.join(where)}
            )
            SELECT
                COUNT(*) AS actual_count,
                MIN(ts_ns) AS first_ts_ns,
                MAX(ts_ns) AS last_ts_ns,
                SUM(
                    CASE
                        WHEN prev_ts_ns IS NOT NULL AND ts_ns - prev_ts_ns != ? THEN 1
                        ELSE 0
                    END
                ) AS bad_gap_count
            FROM ordered
        """
        if _use_pool(self.db_path):
            rows = _pool().execute(sql, params)
            row = rows[0] if rows else None
        else:
            with duckdb.connect(self.db_path) as conn:
                row = conn.execute(sql, params).fetchone()

        actual = int(row[0]) if row and row[0] is not None else 0
        first_ts_ns = row[1] if row and len(row) > 1 else None
        last_ts_ns = row[2] if row and len(row) > 2 else None
        bad_gap_count = int(row[3]) if row and len(row) > 3 and row[3] is not None else 0
        covered = (
            actual == expected
            and first_ts_ns == expected_first_ns
            and last_ts_ns == expected_last_ns
            and bad_gap_count == 0
        )
        return {
            "exchange": exchange,
            "product_id": product_id,
            "subscription_key": subscription_key,
            "covered": covered,
            "expected_count": expected,
            "actual_count": actual,
            "coverage_ratio": min(1.0, actual / expected) if expected > 0 else 1.0,
            "bad_gap_count": bad_gap_count,
            "first_timestamp": None if first_ts_ns is None else _parse_candle_timestamp(first_ts_ns),
            "last_timestamp": None if last_ts_ns is None else _parse_candle_timestamp(last_ts_ns),
        }

    def verify_bag_contiguous_coverage(
        self,
        subscriptions: List[Dict[str, str]],
        start: datetime,
        end: datetime,
        granularity: str = "300",
    ) -> List[Dict[str, object]]:
        statuses: List[Dict[str, object]] = []
        for sub in _dedupe_subscriptions(subscriptions):
            statuses.append(
                self.range_coverage_status(
                    sub["product_id"],
                    start,
                    end,
                    granularity,
                    exchange=sub["exchange"],
                )
            )
        return statuses

    def persist_bag_window_status(
        self,
        subscriptions: List[Dict[str, str]],
        statuses: List[Dict[str, object]],
        start: datetime,
        end: datetime,
        granularity: str = "300",
        min_coverage_ratio: float = 0.0,
        bag_name: Optional[str] = None,
    ) -> Dict[str, str]:
        deduped = _dedupe_subscriptions(subscriptions)
        if not deduped:
            raise ValueError("Cannot persist an empty bag")
        bag_id = _bag_definition_id(deduped)
        bag_window_id = _bag_window_id(
            deduped,
            start,
            end,
            granularity,
            min_coverage_ratio=min_coverage_ratio,
        )
        safe_bag_name = bag_name or bag_id

        member_values_sql = ", ".join(
            f"({_sql_escape(bag_id)}, {_sql_escape(sub['exchange'])}, {_sql_escape(sub['product_id'])})"
            for sub in deduped
        )
        status_values_sql = ", ".join(
            "("
            f"{_sql_escape(bag_window_id)}, "
            f"{_sql_escape(status['exchange'])}, "
            f"{_sql_escape(status['product_id'])}, "
            f"{_sql_escape(bool(status['covered']))}, "
            f"{int(status['expected_count'])}, "
            f"{int(status['actual_count'])}, "
            f"{float(status['coverage_ratio'])}, "
            f"{float(min_coverage_ratio)}, "
            f"{_sql_escape(float(status['coverage_ratio']) >= float(min_coverage_ratio))}, "
            f"{int(status['bad_gap_count'])}, "
            f"{_sql_escape(status['first_timestamp'])}, "
            f"{_sql_escape(status['last_timestamp'])}"
            ")"
            for status in statuses
        )
        stmts = [
            f"INSERT OR REPLACE INTO bags (bag_id, bag_name) VALUES ({_sql_escape(bag_id)}, {_sql_escape(safe_bag_name)})",
            f"INSERT OR REPLACE INTO bag_members (bag_id, exchange, product_id) VALUES {member_values_sql}",
            (
                "INSERT OR REPLACE INTO bag_windows "
                "(bag_window_id, bag_id, window_start, window_end, granularity, min_coverage_ratio) "
                f"VALUES ({_sql_escape(bag_window_id)}, {_sql_escape(bag_id)}, {_sql_escape(start)}, "
                f"{_sql_escape(end)}, {_sql_escape(granularity)}, {_sql_escape(float(min_coverage_ratio))})"
            ),
        ]
        if status_values_sql:
            stmts.append(
                "INSERT OR REPLACE INTO bag_member_status "
                "("
                "bag_window_id, exchange, product_id, covered, expected_count, actual_count, "
                "coverage_ratio, min_coverage_ratio, meets_threshold, bad_gap_count, "
                "first_timestamp, last_timestamp"
                f") VALUES {status_values_sql}"
            )

        if _use_pool(self.db_path):
            p = _pool()
            for stmt in stmts:
                p.execute(stmt)
        else:
            with _db_lock:
                with duckdb.connect(self.db_path) as conn:
                    for stmt in stmts:
                        conn.execute(stmt)
        return {"bag_id": bag_id, "bag_window_id": bag_window_id}

    def materialize_bag_surface(
        self,
        subscriptions: List[Dict[str, str]],
        start: datetime,
        end: datetime,
        granularity: str = "300",
        surface_name: Optional[str] = None,
    ) -> str:
        deduped = _dedupe_subscriptions(subscriptions)
        if not deduped:
            raise ValueError("Cannot materialize a bag surface with no subscriptions")
        surface_name = surface_name or _bag_surface_name(deduped, start, end, granularity)
        values_sql = ", ".join(
            f"({_sql_escape(sub['exchange'])}, {_sql_escape(sub['product_id'])})"
            for sub in deduped
        )
        sql = f"""
            CREATE OR REPLACE VIEW {surface_name} AS
            WITH bag(exchange, product_id) AS (
                VALUES {values_sql}
            )
            SELECT
                c.exchange,
                c.product_id,
                c.timestamp,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.granularity
            FROM candles AS c
            INNER JOIN bag AS b
                ON c.exchange = b.exchange
               AND c.product_id = b.product_id
            WHERE c.granularity = {_sql_escape(granularity)}
              AND c.timestamp >= {_sql_escape(start)}
              AND c.timestamp < {_sql_escape(end)}
            ORDER BY c.exchange, c.product_id, c.timestamp
        """
        if _use_pool(self.db_path):
            _pool().execute(sql)
        else:
            with _db_lock:
                with duckdb.connect(self.db_path) as conn:
                    conn.execute(sql)
        return surface_name

    def materialize_bag_surface_for_window(
        self,
        bag_window_id: str,
        surface_name: Optional[str] = None,
    ) -> str:
        surface_name = surface_name or _bag_surface_view_name_for_window(bag_window_id)
        sql = f"""
            CREATE OR REPLACE VIEW {surface_name} AS
            SELECT
                c.exchange,
                c.product_id,
                c.timestamp,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.granularity
            FROM candles AS c
            JOIN bag_member_status AS bms
              ON bms.exchange = c.exchange
             AND bms.product_id = c.product_id
             AND bms.bag_window_id = {_sql_escape(bag_window_id)}
            JOIN bag_windows AS bw
              ON bw.bag_window_id = bms.bag_window_id
            WHERE bms.covered
              AND bms.meets_threshold
              AND c.granularity = bw.granularity
              AND c.timestamp >= bw.window_start
              AND c.timestamp < bw.window_end
            ORDER BY c.exchange, c.product_id, c.timestamp
        """
        if _use_pool(self.db_path):
            _pool().execute(sql)
        else:
            with _db_lock:
                with duckdb.connect(self.db_path) as conn:
                    conn.execute(sql)
        return surface_name

    def materialize_bag_thresholds_view(
        self,
        statuses: List[Dict[str, object]],
        start: datetime,
        end: datetime,
        granularity: str = "300",
        min_coverage_ratio: float = 0.0,
        view_name: Optional[str] = None,
    ) -> str:
        subscriptions = [
            {"exchange": str(status["exchange"]), "product_id": str(status["product_id"])}
            for status in statuses
        ]
        deduped = _dedupe_subscriptions(subscriptions)
        if not deduped:
            raise ValueError("Cannot materialize bag thresholds view with no subscriptions")
        view_name = view_name or _bag_thresholds_view_name(deduped, start, end, granularity)
        rows_sql = ", ".join(
            "("
            f"{_sql_escape(status['exchange'])}, "
            f"{_sql_escape(status['product_id'])}, "
            f"{_sql_escape(bool(status['covered']))}, "
            f"{int(status['expected_count'])}, "
            f"{int(status['actual_count'])}, "
            f"{float(status['coverage_ratio'])}, "
            f"{int(status['bad_gap_count'])}, "
            f"{_sql_escape(status['first_timestamp'])}, "
            f"{_sql_escape(status['last_timestamp'])}"
            ")"
            for status in statuses
        )
        sql = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                exchange,
                product_id,
                covered,
                expected_count,
                actual_count,
                coverage_ratio,
                {_sql_escape(float(min_coverage_ratio))}::DOUBLE AS min_coverage_ratio,
                (coverage_ratio >= {_sql_escape(float(min_coverage_ratio))}::DOUBLE) AS meets_threshold,
                bad_gap_count,
                first_timestamp,
                last_timestamp,
                {_sql_escape(start)}::TIMESTAMP AS window_start,
                {_sql_escape(end)}::TIMESTAMP AS window_end,
                {_sql_escape(granularity)}::VARCHAR AS granularity
            FROM (
                VALUES {rows_sql}
            ) AS t(
                exchange,
                product_id,
                covered,
                expected_count,
                actual_count,
                coverage_ratio,
                bad_gap_count,
                first_timestamp,
                last_timestamp
            )
            ORDER BY exchange, product_id
        """
        if _use_pool(self.db_path):
            _pool().execute(sql)
        else:
            with _db_lock:
                with duckdb.connect(self.db_path) as conn:
                    conn.execute(sql)
        return view_name

    def read_bag_surface(self, surface_name: str) -> pd.DataFrame:
        sql = f"SELECT * FROM {surface_name} ORDER BY exchange, product_id, timestamp"
        columns = [
            "exchange",
            "product_id",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "granularity",
        ]
        if _use_pool(self.db_path):
            rows = _pool().execute(sql)
            if not rows:
                return pd.DataFrame(columns=columns)
            df = pd.DataFrame(rows, columns=columns)
        else:
            with duckdb.connect(self.db_path) as conn:
                df = conn.execute(sql).df()
        if df.empty:
            return df
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df[df["timestamp"] <= _max_reasonable_future()]
        df = df.drop_duplicates(
            subset=["exchange", "product_id", "timestamp", "granularity"],
            keep="last",
        )
        return df.sort_values(["exchange", "product_id", "timestamp"]).reset_index(drop=True)

    def _count_candles_in_range(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: Optional[str] = None,
    ) -> int:
        status = self.range_coverage_status(
            product_id,
            start,
            end,
            granularity,
            exchange=exchange,
        )
        return int(status["actual_count"])

    def _range_is_fully_cached(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: Optional[str] = None,
    ) -> bool:
        status = self.range_coverage_status(
            product_id,
            start,
            end,
            granularity,
            exchange=exchange,
        )
        return bool(status["covered"])

    def _backfill_plan(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: Optional[str] = None,
    ) -> List[Tuple[datetime, datetime]]:
        status = self.range_coverage_status(
            product_id,
            start,
            end,
            granularity,
            exchange=exchange,
        )
        if bool(status["covered"]):
            return []

        expected_first_ns, expected_last_ns, expected = _expected_bucket_bounds(start, end, granularity)
        if expected <= 0:
            return []

        first_ts = status["first_timestamp"]
        last_ts = status["last_timestamp"]
        gran_sec = _granularity_seconds(granularity)

        if first_ts is None or last_ts is None:
            return [(start, end)]

        if int(status["bad_gap_count"]) > 0:
            return [(start, end)]

        ranges: List[Tuple[datetime, datetime]] = []
        expected_first = _parse_candle_timestamp(expected_first_ns).to_pydatetime()
        expected_last = _parse_candle_timestamp(expected_last_ns).to_pydatetime()
        first_dt = pd.Timestamp(first_ts).to_pydatetime()
        last_dt = pd.Timestamp(last_ts).to_pydatetime()

        if first_dt > expected_first:
            ranges.append((start, first_dt))

        tail_start = last_dt + timedelta(seconds=gran_sec)
        tail_end = expected_last + timedelta(seconds=gran_sec)
        if tail_start < tail_end:
            ranges.append((tail_start, tail_end))

        return [(window_start, window_end) for window_start, window_end in ranges if window_start < window_end]

    def _fetch_chunk(
        self,
        product_id: str,
        exchange: str,
        chunk_start: datetime,
        chunk_end: datetime,
        api_gran: str,
        granularity: str,
        protocol_label: str = "HTTP backfill",
    ) -> int:
        try:
            self._bucket.acquire()
            resp = self.client.get_public_candles(
                product_id=product_id,
                start=str(_epoch_seconds_utc(chunk_start)),
                end=str(_epoch_seconds_utc(chunk_end)),
                granularity=api_gran,
            )
            if not (hasattr(resp, 'candles') and resp.candles):
                return 0
            rows = [{
                'exchange': exchange,
                'product_id': product_id,
                'timestamp': _parse_candle_timestamp(c.start),
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
            print(f"  [{protocol_label}][{exchange}][{product_id}] chunk error: {e}")
            return 0

    def _fetch_and_save(
        self,
        product_id: str,
        exchange: str,
        start: datetime,
        end: datetime,
        granularity: str,
        force: bool = False,
        protocol_label: str = "HTTP backfill",
    ):
        api_gran = {
            "60": "ONE_MINUTE", "300": "FIVE_MINUTE", "900": "FIFTEEN_MINUTE",
            "3600": "ONE_HOUR", "21600": "SIX_HOUR", "86400": "ONE_DAY",
        }.get(granularity, "FIVE_MINUTE")

        aligned_start = _floor_time_to_granularity(start, granularity)
        aligned_end = _ceil_time_to_granularity(end, granularity)
        gran_sec = int(granularity)
        chunk_sec = CHUNK_CANDLES * gran_sec

        chunks = []
        cur = aligned_start
        while cur < aligned_end:
            chunks.append((cur, min(cur + timedelta(seconds=chunk_sec), aligned_end)))
            cur += timedelta(seconds=chunk_sec)

        total = 0
        skipped = 0
        i = 0
        while i < len(chunks):
            chunk_start, chunk_end = chunks[i]
            if not force and self._range_is_fully_cached(
                product_id,
                chunk_start,
                chunk_end,
                granularity,
                exchange=exchange,
            ):
                skipped += 1
                i += 1
                continue
            result = self._fetch_chunk(
                product_id,
                exchange,
                chunk_start,
                chunk_end,
                api_gran,
                granularity,
                protocol_label=protocol_label,
            )
            if result == -1:
                with self._concurrency_lock:
                    self._concurrency = max(1, self._concurrency - 2)
                    print(
                        f"  [{protocol_label}][{exchange}][{product_id}] "
                        f"429 hit — concurrency → {self._concurrency}, sleeping 15s"
                    )
                time.sleep(15)
            else:
                total += result
                i += 1

        print(
            f"  [{protocol_label}][{exchange}][{product_id}] {total} candles saved, "
            f"{skipped}/{len(chunks)} chunks skipped (cached), "
            f"window={aligned_start.isoformat()} -> {aligned_end.isoformat()}"
        )

    def backfill_http(
        self,
        pairs: List[str],
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: str = "coinbase",
    ):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not pairs:
            return
        if exchange != "coinbase":
            raise NotImplementedError(f"HTTP backfill is only implemented for coinbase, not {exchange}")
        plans: List[Tuple[str, datetime, datetime]] = []
        skipped_pairs = 0
        full_window_pairs = 0
        partial_window_pairs = 0
        for pid in pairs:
            ranges = self._backfill_plan(
                pid,
                start,
                end,
                granularity=granularity,
                exchange=exchange,
            )
            if not ranges:
                skipped_pairs += 1
                continue
            if len(ranges) == 1 and ranges[0][0] == start and ranges[0][1] == end:
                full_window_pairs += 1
            else:
                partial_window_pairs += 1
            for window_start, window_end in ranges:
                plans.append((pid, window_start, window_end))

        print(
            f"[HTTP backfill] {len(pairs)} pairs, "
            f"{start.isoformat()} -> {end.isoformat()}, granularity={granularity}, "
            f"skip={skipped_pairs}, partial={partial_window_pairs}, full={full_window_pairs}"
        )
        if not plans:
            return

        def fetch_one(plan: Tuple[str, datetime, datetime]) -> Tuple[str, datetime, datetime]:
            pid, window_start, window_end = plan
            self._fetch_and_save(
                pid,
                exchange,
                window_start,
                window_end,
                granularity,
                protocol_label="HTTP backfill",
            )
            return plan

        remaining = list(plans)
        while remaining:
            with self._concurrency_lock:
                workers = self._concurrency
            batch = remaining[:workers * 2]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(fetch_one, plan): plan for plan in batch}
                done: List[Tuple[str, datetime, datetime]] = []
                for future in as_completed(futures):
                    pid, window_start, window_end = futures[future]
                    try:
                        future.result()
                        done.append((pid, window_start, window_end))
                        print(
                            f"  ✓ [HTTP backfill][{exchange}][{pid}] "
                            f"{window_start.isoformat()} -> {window_end.isoformat()} "
                            f"({len(done)}/{len(batch)})"
                        )
                    except Exception as e:
                        print(
                            f"  ✗ [HTTP backfill][{exchange}][{pid}] "
                            f"{window_start.isoformat()} -> {window_end.isoformat()}: {e}"
                        )
                        done.append((pid, window_start, window_end))
            remaining = [plan for plan in remaining if plan not in done]

    def repair_http_overlap(
        self,
        pairs: List[str],
        granularity: str = "300",
        overlap_candles: int = 6,
        end: Optional[datetime] = None,
        exchange: str = "coinbase",
    ):
        if not pairs:
            return
        if exchange != "coinbase":
            raise NotImplementedError(f"HTTP repair is only implemented for coinbase, not {exchange}")
        gran_sec = int(granularity)
        repair_end = _floor_time_to_granularity(end or _utc_now_naive(), granularity)
        floor = repair_end - timedelta(seconds=gran_sec * max(overlap_candles, 1))

        for pid in pairs:
            latest = self.latest_timestamp(pid, granularity, exchange=exchange)
            start = floor if latest is None else min(
                latest - timedelta(seconds=gran_sec * max(overlap_candles, 1)),
                floor,
            )
            if start >= repair_end:
                continue
            self._fetch_and_save(
                pid,
                exchange,
                start,
                repair_end,
                granularity,
                force=True,
                protocol_label="HTTP repair",
            )

    def ws_snapshot(self, pairs: List[str], granularity: str = "300", exchange: str = "coinbase"):
        from coinbase.websocket import WSClient

        if exchange != "coinbase":
            raise NotImplementedError(f"Websocket snapshot is only implemented for coinbase, not {exchange}")
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
                            'exchange': exchange,
                            'product_id': pid,
                            'timestamp': _parse_candle_timestamp(c['start']),
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

    def stream_live(
        self,
        pairs: List[str],
        granularity: str = "300",
        stop_event: Optional[threading.Event] = None,
        on_rows: Optional[Callable[[pd.DataFrame], None]] = None,
        repair_every_seconds: int = 30,
        overlap_candles: int = 6,
        idle_restart_seconds: int = 90,
        exchange: str = "coinbase",
    ):
        from coinbase.websocket import WSClient

        if not pairs:
            return
        if exchange != "coinbase":
            raise NotImplementedError(f"Websocket live stream is only implemented for coinbase, not {exchange}")

        stop_event = stop_event or threading.Event()
        pair_set = set(pairs)
        state = {
            "last_activity": time.monotonic(),
            "last_repair": 0.0,
        }

        def _decode_rows(msg) -> pd.DataFrame:
            try:
                if isinstance(msg, str):
                    data = json.loads(msg)
                elif hasattr(msg, "to_dict"):
                    data = msg.to_dict()
                else:
                    data = msg
            except Exception:
                return pd.DataFrame()

            if not isinstance(data, dict):
                return pd.DataFrame()

            rows = []
            for event in data.get("events", []):
                candles = event.get("candles", [])
                for candle in candles:
                    pid = candle.get("product_id")
                    if pid not in pair_set:
                        continue
                    try:
                        rows.append(
                            {
                                "exchange": exchange,
                                "product_id": pid,
                                "timestamp": _parse_candle_timestamp(candle["start"]),
                                "open": float(candle["open"]),
                                "high": float(candle["high"]),
                                "low": float(candle["low"]),
                                "close": float(candle["close"]),
                                "volume": float(candle["volume"]),
                                "granularity": granularity,
                            }
                        )
                    except Exception:
                        continue
            return pd.DataFrame(rows)

        while not stop_event.is_set():
            client = None
            try:
                self.repair_http_overlap(
                    pairs,
                    granularity=granularity,
                    overlap_candles=overlap_candles,
                    exchange=exchange,
                )
                state["last_repair"] = time.monotonic()

                def on_message(msg):
                    df = _decode_rows(msg)
                    if df.empty:
                        return
                    self.save_candles(df)
                    state["last_activity"] = time.monotonic()
                    if on_rows is not None:
                        try:
                            on_rows(df)
                        except Exception:
                            pass

                client = WSClient(api_key=None, api_secret=None, on_message=on_message)
                client.open()
                client.candles(pairs)
                print(f"[WS live] subscribed to {len(pairs)} pairs")

                while not stop_event.is_set():
                    now = time.monotonic()
                    if now - state["last_repair"] >= repair_every_seconds:
                        self.repair_http_overlap(
                            pairs,
                            granularity=granularity,
                            overlap_candles=overlap_candles,
                            exchange=exchange,
                        )
                        state["last_repair"] = now
                    if now - state["last_activity"] >= idle_restart_seconds:
                        print("[WS live] idle timeout; restarting websocket after REST repair")
                        break
                    time.sleep(1.0)
            except Exception as exc:
                print(f"[WS live] error: {exc}")
                time.sleep(3.0)
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass

        print("[WS live] stopped")

    def prefetch_all(
        self,
        pairs: List[str],
        start: datetime,
        end: datetime,
        granularity: str = "300",
        exchange: str = "coinbase",
        protocol_label: str = "HTTP backfill",
    ):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch_one(pid):
            self._fetch_and_save(
                pid,
                exchange,
                start,
                end,
                granularity,
                protocol_label=protocol_label,
            )
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
                        print(f"  ✓ [{protocol_label}][{exchange}][{pid}] ({len(done)}/{len(batch)})")
                    except Exception as e:
                        print(f"  ✗ [{protocol_label}][{exchange}][{pid}]: {e}")
                        done.append(pid)
            remaining = [p for p in remaining if p not in done]

    def prefetch_all_async(
        self,
        pairs: List[str],
        start: datetime,
        end: datetime,
        granularity: str = "300",
        name: Optional[str] = None,
        exchange: str = "coinbase",
        protocol_label: str = "HTTP backfill",
    ) -> threading.Thread:
        thread = threading.Thread(
            target=self.prefetch_all,
            args=(list(pairs), start, end, granularity),
            kwargs={"exchange": exchange, "protocol_label": protocol_label},
            name=name or f"prefetch-{granularity}",
            daemon=True,
        )
        thread.start()
        return thread

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
        cutoff_ts = pd.Timestamp(_utc_now_naive()) - pd.Timedelta(days=3 * 365)
        cutoff_ms = int(cutoff_ts.timestamp() * 1000)

        for product_id in pairs:
            base, quote = product_id.split("-", 1)
            binance_quote = "USDT" if quote == "USD" else quote
            if base == binance_quote:
                continue  # e.g. USDT-USDT — skip

            # Skip if DuckDB already has >1000 bars in the 3-year window
            with duckdb.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM candles WHERE exchange=? AND product_id=? AND granularity=? AND timestamp >= ?",
                    ["binance", product_id, granularity, cutoff_ts]
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
                resampled["exchange"] = "binance"
                resampled["product_id"] = product_id
                resampled["granularity"] = granularity
                out = resampled[["exchange", "product_id", "timestamp", "open", "high", "low", "close", "volume", "granularity"]]
                self.save_candles(out)
                yrs = len(out) * gran_sec / (365.25 * 86400)
                print(f"[Binance] {product_id}: {len(out):,} bars imported ({yrs:.1f}yr)")
                imported.append(product_id)
            except Exception as e:
                print(f"[Binance] {product_id}: {e}")

        return imported

    def save_edge_params(self, params: dict):
        """Upsert edge params. params = {edge_str: {curvature, y_depth, x_pixels, fee_rate}}"""
        if _use_pool(self.db_path):
            p = _pool()
            for edge, ep in params.items():
                p.execute(
                    """INSERT INTO edge_params (edge, curvature, y_depth, x_pixels, fee_rate, updated_at)
                       VALUES (?, ?, ?, ?, ?, now())
                       ON CONFLICT (edge) DO UPDATE SET
                           curvature=excluded.curvature,
                           y_depth=excluded.y_depth,
                           x_pixels=excluded.x_pixels,
                           fee_rate=excluded.fee_rate,
                           updated_at=excluded.updated_at""",
                    [edge, ep['curvature'], ep['y_depth'], ep['x_pixels'], ep['fee_rate']],
                )
        else:
            with _db_lock:
                with duckdb.connect(self.db_path) as conn:
                    for edge, ep in params.items():
                        conn.execute("""
                            INSERT INTO edge_params (edge, curvature, y_depth, x_pixels, fee_rate, updated_at)
                            VALUES (?, ?, ?, ?, ?, now())
                            ON CONFLICT (edge) DO UPDATE SET
                                curvature=excluded.curvature,
                                y_depth=excluded.y_depth,
                                x_pixels=excluded.x_pixels,
                                fee_rate=excluded.fee_rate,
                                updated_at=excluded.updated_at
                        """, [edge, ep['curvature'], ep['y_depth'], ep['x_pixels'], ep['fee_rate']])

    def load_edge_params(self) -> dict:
        """Returns {edge_str: {curvature, y_depth, x_pixels, fee_rate}}"""
        if _use_pool(self.db_path):
            rows = _pool().execute(
                "SELECT edge, curvature, y_depth, x_pixels, fee_rate FROM edge_params"
            )
        else:
            with duckdb.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT edge, curvature, y_depth, x_pixels, fee_rate FROM edge_params"
                ).fetchall()
        return {r[0]: {'curvature': r[1], 'y_depth': r[2], 'x_pixels': r[3], 'fee_rate': r[4]} for r in rows}
