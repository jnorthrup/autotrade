"""
cache.py - Data ingestion, DuckDB pooling, Graph building, Bag IO, and Gap Repair.
Consolidates config.py, pool_client.py, bag_io.py, candle_cache.py, and coin_graph.py.
"""

import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd
import duckdb
from coinbase.rest import RESTClient
from dotenv import load_dotenv

load_dotenv()


# --- Config ---
class Config:
    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
    COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
    DB_PATH = Path(os.getenv("DB_PATH", "candles.duckdb")).resolve()
    BAG_PATH = Path(os.getenv("BAG_PATH", "bag.json")).resolve()
    DEFAULT_GRANULARITY = os.getenv("DEFAULT_GRANULARITY", "300")
    USE_WS_ONLY = os.getenv("USE_WS_ONLY", "true").lower() in ["1", "true", "yes"]


# --- Pool Client ---
DEFAULT_SOCKET = "/tmp/duckdb_pool.sock"
POOL_BINARY = Path(__file__).parent / "pool_server.py"
POOL_AUTOSTART_LOG = Path(__file__).parent / "logs" / "duckdb_pool.autostart.log"
_pool_process = None
_pool_log_handle = None


def _sql_escape(val) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (pd.Timestamp, datetime)):
        return f"'{val.isoformat()}'"
    return f"'{str(val).replace(chr(39), chr(39) + chr(39))}'"


def _interpolate_params(sql: str, params: Optional[List[Any]] = None) -> str:
    if not params:
        return sql
    parts = sql.split("?")
    return parts[0] + "".join(
        _sql_escape(params[i]) + part for i, part in enumerate(parts[1:])
    )


class PoolClient:
    def __init__(self, socket_path: Optional[str] = None):
        self.socket_path = socket_path or DEFAULT_SOCKET

    def _call(self, req: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
            sock.settimeout(300)
            sock.sendall((json.dumps(req) + "\n").encode())

            with sock.makefile("r", encoding="utf-8") as f:
                line = f.readline()
                return json.loads(line)
        finally:
            sock.close()

    def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[List[Any]]:
        resp = self._call({"query": _interpolate_params(sql, params)})
        if resp.get("error"):
            raise RuntimeError(f"Pool error: {resp['error']}")
        return resp.get("rows", [])

    def execute_df(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        rows = self.execute(sql, params)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def ping(self) -> bool:
        try:
            return self.execute("SELECT 1") == [[1]]
        except Exception:
            return False


def pool_is_running(socket_path: Optional[str] = None) -> bool:
    sp = socket_path or DEFAULT_SOCKET
    return os.path.exists(sp) and PoolClient(sp).ping()


def ensure_pool_running(
    db_path: str, socket_path: Optional[str] = None, startup_timeout: float = 10.0
) -> PoolClient:
    sp = socket_path or DEFAULT_SOCKET
    if pool_is_running(sp):
        return PoolClient(sp)
    global _pool_process, _pool_log_handle
    if not POOL_BINARY.exists():
        raise RuntimeError(f"Pool binary not found at {POOL_BINARY}")
    if os.path.exists(sp):
        try:
            os.remove(sp)
        except OSError:
            pass
    if _pool_process is None or _pool_process.poll() is not None:
        POOL_AUTOSTART_LOG.parent.mkdir(parents=True, exist_ok=True)
        _pool_log_handle = (
            open(POOL_AUTOSTART_LOG, "ab", buffering=0)
            if not _pool_log_handle
            else _pool_log_handle
        )
        _pool_process = subprocess.Popen(
            [str(POOL_BINARY), str(Path(db_path).resolve()), "--socket", sp],
            stdin=subprocess.DEVNULL,
            stdout=_pool_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if pool_is_running(sp):
            return PoolClient(sp)
        time.sleep(0.1)
    raise RuntimeError("DuckDB pool failed to start")


def _use_pool(db_path: Optional[str] = None) -> bool:
    try:
        if not pool_is_running():
            return False
        return (
            True
            if db_path is None
            else Path(db_path).resolve() == Path(Config.DB_PATH).resolve()
        )
    except Exception:
        return False


def _pool() -> PoolClient:
    return PoolClient(DEFAULT_SOCKET)


def _use_pool_for_db(db_path: str) -> bool:
    """Compatibility wrapper used by other scripts expecting _use_pool_for_db.

    Returns True when a pool is running and the given db_path matches the
    configured DB_PATH for this project.
    """
    try:
        return _use_pool(db_path)
    except Exception:
        return False


# --- Time Utils ---
def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _utc_isoformat(value: datetime) -> str:
    ts = pd.Timestamp(value)
    return (
        (ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC"))
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_candle_timestamp(raw_value) -> pd.Timestamp:
    if isinstance(raw_value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(raw_value)
    elif isinstance(raw_value, str) and not raw_value.strip().isdigit():
        ts = pd.Timestamp(raw_value)
    else:
        v = abs(int(raw_value))
        ts = pd.to_datetime(
            int(raw_value),
            unit="ns"
            if v >= 10**18
            else "us"
            if v >= 10**15
            else "ms"
            if v >= 10**12
            else "s",
        )
    return pd.Timestamp(ts.tz_convert(None) if ts.tzinfo else ts)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _granularity_seconds(granularity: str) -> int:
    return max(1, int(granularity))


def _floor_time_to_granularity(value: datetime, granularity: str) -> datetime:
    gran_ns = _granularity_seconds(granularity) * 1_000_000_000
    return pd.Timestamp(
        (int(pd.Timestamp(value).value) // gran_ns) * gran_ns
    ).to_pydatetime()


def _ceil_time_to_granularity(value: datetime, granularity: str) -> datetime:
    gran_ns = _granularity_seconds(granularity) * 1_000_000_000
    v = int(pd.Timestamp(value).value)
    return (
        pd.Timestamp(v).to_pydatetime()
        if v % gran_ns == 0
        else pd.Timestamp(((v // gran_ns) + 1) * gran_ns).to_pydatetime()
    )


# --- Bag IO ---
EdgeKey = Tuple[str, ...]


def _parse_edge(edge: EdgeKey) -> Tuple[Optional[str], str, str]:
    if len(edge) == 2:
        return None, str(edge[0]), str(edge[1])
    if len(edge) == 3:
        return str(edge[0]), str(edge[1]), str(edge[2])
    raise ValueError(f"Unsupported edge shape: {edge!r}")


def _dedupe_subscriptions(subscriptions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return list(
        {
            (
                str(s.get("exchange", "")).strip(),
                str(s.get("product_id", "")).strip(),
            ): {
                "exchange": str(s.get("exchange", "")).strip(),
                "product_id": str(s.get("product_id", "")).strip(),
            }
            for s in (subscriptions or [])
            if str(s.get("exchange", "")).strip()
            and "-" in str(s.get("product_id", "")).strip()
        }.values()
    )


def _load_explicit_bag_subscriptions(bag_path: str) -> List[Dict[str, str]]:
    with open(bag_path, "r") as f:
        return _dedupe_subscriptions(json.load(f))


def build_bag_model_frames(
    graph,
    *,
    bar_idx: int,
    edge_names: Sequence[EdgeKey],
    node_to_idx: Mapping[str, int],
    value_asset: str = "USD",
    edge_fisheyes: Optional[Mapping[EdgeKey, Sequence[float]]] = None,
    edge_carries: Optional[Mapping[EdgeKey, Any]] = None,
    free_qty: Optional[Any] = None,
    reserved_qty: Optional[Any] = None,
    route_discount: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if bar_idx < 0 or bar_idx >= len(getattr(graph, "common_timestamps", [])):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ts = graph.common_timestamps[bar_idx]
    rows = []
    for edge in edge_names:
        df = graph.edges.get(edge)
        if df is None or ts not in df.index:
            continue
        row = df.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        exchange, base_asset, quote_asset = _parse_edge(edge)
        close = _finite_float(row.get("close", 0.0), 0.0)
        if getattr(graph, "edge_is_inverted", {}).get(edge, False) and close > 0:
            close = 1.0 / close
        open_price = _finite_float(row.get("open", close), close)
        if getattr(graph, "edge_is_inverted", {}).get(edge, False) and open_price > 0:
            open_price = 1.0 / open_price
        price_now = close
        edge_discount = (
            float((route_discount or {}).get(edge, 0.0))
            if isinstance(route_discount, Mapping)
            else float(route_discount or 0.0)
        )
        fee_rate = float(getattr(graph, "fee_rate", 0.0))
        edge_state = getattr(graph, "edge_state", {}).get(edge)
        payload = {
            "ts": ts,
            "exchange": exchange,
            "base_idx": node_to_idx[base_asset],
            "quote_idx": node_to_idx[quote_asset],
            "base_asset": base_asset,
            "quote_asset": quote_asset,
            "price_now": price_now,
            "edge_velocity": float(getattr(edge_state, "velocity", 0.0)),
            "edge_ptt": float(getattr(edge_state, "ptt", 0.0)),
            "edge_stop": float(getattr(edge_state, "stop", 0.0)),
            "fee_rate": fee_rate,
            "route_discount": edge_discount,
            "edge_discount_total": fee_rate + edge_discount,
            "discounted_price": price_now * math.exp(-(fee_rate + edge_discount))
            if price_now > 0
            else 0.0,
        }
        for idx, value in enumerate((edge_fisheyes or {}).get(edge, ())):
            payload[f"fisheye_{idx}"] = float(value)
        c = (edge_carries or {}).get(edge)
        if c and getattr(c, "z_H", None) is not None:
            for idx, value in enumerate(
                np.asarray(c.z_H.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            ):
                payload[f"carry_h_{idx}"] = float(value)
        if c and getattr(c, "z_L", None) is not None:
            for idx, value in enumerate(
                np.asarray(c.z_L.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            ):
                payload[f"carry_l_{idx}"] = float(value)
        rows.append(payload)
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pair_state_df = (
        pd.DataFrame(rows).sort_values(["base_idx", "quote_idx"]).reset_index(drop=True)
    )

    node_names = [n for n, _ in sorted(node_to_idx.items(), key=lambda i: i[1])]
    value_idx = node_to_idx.get(value_asset)
    best_log_value = {i: float("-inf") for i in range(len(node_names))}
    if value_idx is not None:
        best_log_value[value_idx] = 0.0
    for _ in range(max(0, len(node_names) - 1)):
        changed = False
        for row in pair_state_df.itertuples(index=False):
            src, dst, rate = (
                int(row.base_idx),
                int(row.quote_idx),
                float(row.discounted_price),
            )
            if (
                rate > 0
                and best_log_value[dst] != float("-inf")
                and math.log(rate) + best_log_value[dst] > best_log_value[src] + 1e-12
            ):
                best_log_value[src] = math.log(rate) + best_log_value[dst]
                changed = True
        if not changed:
            break

    node_rows = []
    for coin_idx, asset in enumerate(node_names):
        # best_log_value stores the sum of log(discounted_price) along the best path
        # from the asset to the value asset. The asset value (price in value_asset
        # terms) is the exponential of that sum. Previous code used a negative
        # exponent here which inverted the price (producing 1/price) causing
        # valuations to appear to not move forward. Use exp(best_log_value).
        root_price = (
            math.exp(best_log_value[coin_idx])
            if best_log_value[coin_idx] != float("-inf")
            else 0.0
        )
        fq = (
            float((free_qty or {}).get(asset, 0.0))
            if isinstance(free_qty, Mapping)
            else 0.0
        )
        rq = (
            float((reserved_qty or {}).get(asset, 0.0))
            if isinstance(reserved_qty, Mapping)
            else 0.0
        )
        node_rows.append(
            {
                "ts": ts,
                "coin_idx": coin_idx,
                "asset": asset,
                "root_price": root_price,
                "free_qty": fq,
                "reserved_qty": rq,
            }
        )
    node_state_df = (
        pd.DataFrame(node_rows).sort_values("coin_idx").reset_index(drop=True)
    )
    return node_state_df, pair_state_df, pair_state_df


def bind_spend_budgets(
    pred_df: pd.DataFrame,
    node_state_df: pd.DataFrame,
    *,
    hi_col: str = "hi",
    lo_col: str = "lo",
    fraction_col: str = "fraction",
) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()
    exec_df = pred_df.copy()
    exec_df["fraction"] = exec_df[fraction_col].clip(lower=0.0, upper=1.0)
    exec_df["polarity"] = np.where(exec_df[hi_col] >= exec_df[lo_col], "hi", "lo")
    exec_df["spend_idx"] = np.where(
        exec_df["polarity"].eq("hi"), exec_df["base_idx"], exec_df["quote_idx"]
    )
    exec_df = exec_df.merge(
        node_state_df[["coin_idx", "asset", "free_qty"]].rename(
            columns={
                "coin_idx": "spend_idx",
                "asset": "spend_asset",
                "free_qty": "available_spend_qty",
            }
        ),
        on="spend_idx",
        how="left",
    )
    exec_df["available_spend_qty"] = exec_df["available_spend_qty"].fillna(0.0)
    exec_df["requested_spend_qty"] = (
        exec_df["fraction"] * exec_df["available_spend_qty"]
    )
    total_claim = exec_df.groupby("spend_idx")["requested_spend_qty"].transform("sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        exec_df["clip_scale"] = np.where(
            total_claim > exec_df["available_spend_qty"],
            np.where(
                total_claim > 0, exec_df["available_spend_qty"] / total_claim, 1.0
            ),
            1.0,
        )
    exec_df["final_spend_qty"] = exec_df["requested_spend_qty"] * exec_df["clip_scale"]
    return exec_df


# --- Candle Cache ---
class _TokenBucket:
    def __init__(self, rate: float):
        self._rate, self._tokens, self._last, self._lock, self._pause_until = (
            rate,
            rate,
            time.monotonic(),
            threading.Lock(),
            0.0,
        )

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                if self._pause_until > now:
                    wait = self._pause_until - now
                else:
                    self._tokens = min(
                        self._rate, self._tokens + (now - self._last) * self._rate
                    )
                    self._last = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)

    def pause(self, seconds: float):
        with self._lock:
            self._pause_until = max(
                self._pause_until, time.monotonic() + max(0.0, seconds)
            )
            self._tokens, self._last = 0.0, self._pause_until


class CandleCache:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Config.DB_PATH)
        self.client = RESTClient(Config.COINBASE_API_KEY, Config.COINBASE_API_SECRET)
        self._init_db()
        self._concurrency, self._concurrency_lock, self._bucket = (
            4,
            threading.Lock(),
            _TokenBucket(3.0),
        )

    def _init_db(self):
        stmts = [
            "CREATE TABLE IF NOT EXISTS candles (exchange VARCHAR, product_id VARCHAR, timestamp TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, granularity VARCHAR, span INTEGER DEFAULT 1, PRIMARY KEY (exchange, product_id, timestamp, granularity))",
            "CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_candles_exchange_time ON candles(exchange, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_candles_exchange_prod_gran ON candles(exchange, product_id, granularity)",
        ]
        if _use_pool(self.db_path):
            p = _pool()
            for s in stmts:
                p.execute(s)
        else:
            with duckdb.connect(self.db_path) as conn:
                for s in stmts:
                    conn.execute(s)

    def bootstrap_database(
        self,
        granularity: Optional[str] = None,
        exchange: Optional[str] = None,
        future_days: int = 1,
    ) -> Dict[str, object]:
        return {
            "db_path": str(Path(self.db_path).resolve()),
            "granularity": granularity,
            "exchange": exchange,
            "normalized_timestamps": 0,
            "purged_future_rows": self.purge_future_candles(
                granularity, future_days, exchange
            ),
            "pool_enabled": bool(_use_pool(self.db_path)),
        }

    def purge_future_candles(
        self,
        granularity: Optional[str] = None,
        future_days: int = 1,
        exchange: Optional[str] = None,
    ) -> int:
        where, params = (
            "timestamp > ?",
            [_utc_now_naive() + timedelta(days=future_days)],
        )
        if granularity:
            where += " AND granularity = ?"
            params.append(granularity)
        if exchange:
            where += " AND exchange = ?"
            params.append(exchange)
        count_sql, delete_sql = (
            f"SELECT COUNT(*) FROM candles WHERE {where}",
            f"DELETE FROM candles WHERE {where}",
        )
        if _use_pool(self.db_path):
            rows = _pool().execute(count_sql, params)
            if rows and rows[0][0]:
                _pool().execute(delete_sql, params)
            return int(rows[0][0]) if rows else 0
        with duckdb.connect(self.db_path) as conn:
            row = conn.execute(count_sql, params).fetchone()
            if row and row[0]:
                conn.execute(delete_sql, params)
            return int(row[0]) if row else 0

    def save_candles(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df[
            df["timestamp"] <= pd.Timestamp(_utc_now_naive()) + pd.Timedelta(days=1)
        ]
        if "span" not in df.columns:
            df["span"] = 1
        df = df.drop_duplicates(
            subset=["exchange", "product_id", "timestamp", "granularity"], keep="last"
        )
        if df.empty:
            return
        if _use_pool(self.db_path):
            os.system("pkill -9 duckdb_pool || true")
            time.sleep(0.5)

        with duckdb.connect(self.db_path) as conn:
            conn.register("df", df)
            conn.execute("INSERT OR REPLACE INTO candles SELECT * FROM df")

    def _get_bag_gaps(
        self,
        subscriptions: List[Dict[str, str]],
        start: datetime,
        end: datetime,
        granularity: str,
    ) -> pd.DataFrame:
        deduped = _dedupe_subscriptions(subscriptions)
        if not deduped:
            return pd.DataFrame(
                columns=["exchange", "product_id", "gap_start", "gap_end"]
            )
        values_sql = ", ".join(
            f"({_sql_escape(sub['exchange'])}, {_sql_escape(sub['product_id'])})"
            for sub in deduped
        )
        sql = f"""
            WITH bag(exchange, product_id) AS (VALUES {values_sql}),
            bounds AS (SELECT {_sql_escape(start)}::TIMESTAMP AS w_start, {_sql_escape(end)}::TIMESTAMP AS w_end, {int(granularity)}::INTEGER AS gran_sec),
            available AS (SELECT c.exchange, c.product_id, c.timestamp AS ts, c.timestamp + INTERVAL (c.span * bnd.gran_sec) SECOND AS ts_end FROM candles c JOIN bag b ON c.exchange = b.exchange AND c.product_id = b.product_id CROSS JOIN bounds bnd WHERE c.granularity = {_sql_escape(granularity)} AND c.timestamp + INTERVAL (c.span * bnd.gran_sec) SECOND > bnd.w_start AND c.timestamp < bnd.w_end),
            endpoints AS (SELECT exchange, product_id, w_start AS ts, w_start AS ts_end FROM bag CROSS JOIN bounds UNION ALL SELECT exchange, product_id, w_end AS ts, w_end AS ts_end FROM bag CROSS JOIN bounds UNION ALL SELECT exchange, product_id, ts, ts_end FROM available),
            ordered AS (SELECT exchange, product_id, ts, ts_end, MAX(ts_end) OVER (PARTITION BY exchange, product_id ORDER BY ts, ts_end DESC ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS max_prev_end FROM endpoints),
            gaps AS (SELECT exchange, product_id, GREATEST(COALESCE(max_prev_end, ts), (SELECT w_start FROM bounds)) AS gap_start, LEAST(ts, (SELECT w_end FROM bounds)) AS gap_end FROM ordered WHERE ts > COALESCE(max_prev_end, ts))
            SELECT exchange, product_id, gap_start, gap_end FROM gaps WHERE gap_start < gap_end ORDER BY gap_start, gap_end, exchange, product_id
        """
        if _use_pool(self.db_path):
            df = pd.DataFrame(
                _pool().execute(sql),
                columns=["exchange", "product_id", "gap_start", "gap_end"],
            )
        else:
            with duckdb.connect(self.db_path) as conn:
                df = conn.execute(sql).df()
        if not df.empty:
            df["gap_start"], df["gap_end"] = (
                pd.to_datetime(df["gap_start"]),
                pd.to_datetime(df["gap_end"]),
            )
        return df

    def backfill_http_exact(
        self,
        plans: List[Tuple[str, datetime, datetime]],
        granularity: str = "300",
        exchange: str = "coinbase",
        protocol_label: str = "HTTP backfill/exact",
    ):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if exchange != "coinbase":
            return
        normalized_plans = sorted(
            {
                (
                    str(pid),
                    _floor_time_to_granularity(start, granularity),
                    _floor_time_to_granularity(end, granularity),
                )
                for pid, start, end in plans
                if start < end
            },
            key=lambda i: (i[1], i[2], i[0]),
        )
        if not normalized_plans:
            return

        def fetch_one(plan):
            pid, w_start, w_end = plan
            try:
                self._bucket.acquire()
                resp = self.client.get_public_candles(
                    product_id=pid,
                    start=str(int(w_start.timestamp())),
                    end=str(int(w_end.timestamp())),
                    granularity={
                        "60": "ONE_MINUTE",
                        "300": "FIVE_MINUTE",
                        "900": "FIFTEEN_MINUTE",
                        "3600": "ONE_HOUR",
                        "21600": "SIX_HOUR",
                        "86400": "ONE_DAY",
                    }.get(granularity, "FIVE_MINUTE"),
                )
                rows = []
                for c in (
                    resp.candles if hasattr(resp, "candles") and resp.candles else []
                ):
                    ts = _parse_candle_timestamp(c.start)
                    if w_start <= ts < w_end:
                        rows.append(
                            {
                                "exchange": exchange,
                                "product_id": pid,
                                "timestamp": ts,
                                "open": float(c.open),
                                "high": float(c.high),
                                "low": float(c.low),
                                "close": float(c.close),
                                "volume": float(c.volume),
                                "granularity": granularity,
                                "span": 1,
                            }
                        )
                if rows:
                    self.save_candles(pd.DataFrame(rows))
            except Exception as e:
                if "429" in str(e):
                    self._bucket.pause(15.0)

        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            for _ in as_completed(
                {pool.submit(fetch_one, plan): plan for plan in normalized_plans}
            ):
                pass

    def repair_bag_drawthrough(
        self,
        subscriptions: List[Dict[str, str]],
        start: datetime,
        end: datetime,
        granularity: str = "300",
        *,
        max_passes: int = 6,
        log_prefix: str = "[Bag repair]",
        label: str = "drawthrough",
    ) -> Dict[str, object]:
        deduped = _dedupe_subscriptions(subscriptions)
        if not deduped:
            return {"status": "empty", "passes": 0, "remaining_fragments": 0}
        window_start, window_end = (
            _floor_time_to_granularity(start, granularity),
            _floor_time_to_granularity(end, granularity),
        )
        if window_start >= window_end:
            return {"status": "invalid_window", "passes": 0, "remaining_fragments": 0}
        previous_signature = ()
        for pass_idx in range(1, max(max_passes, 1) + 1):
            gaps_df = self._get_bag_gaps(deduped, window_start, window_end, granularity)
            if gaps_df.empty:
                print(
                    f"{log_prefix} {label} perfected window={_utc_isoformat(window_start)} -> {_utc_isoformat(window_end)}"
                )
                return {
                    "status": "perfected",
                    "passes": pass_idx - 1 if pass_idx > 1 else 0,
                    "remaining_fragments": 0,
                }
            signature = tuple(
                (
                    str(r.exchange),
                    str(r.product_id),
                    int(r.gap_start.timestamp()),
                    int(r.gap_end.timestamp()),
                )
                for r in gaps_df.itertuples()
            )
            if signature == previous_signature:
                print(
                    f"{log_prefix} {label} stalled remaining_fragments={len(gaps_df)}"
                )
                return {
                    "status": "stalled",
                    "passes": pass_idx,
                    "remaining_fragments": len(gaps_df),
                }
            previous_signature = signature
            print(
                f"{log_prefix} {label} pass={pass_idx}/{max(max_passes, 1)} fragments={len(gaps_df)} window={_utc_isoformat(window_start)} -> {_utc_isoformat(window_end)}"
            )
            for exchange, group in gaps_df.groupby("exchange"):
                self.backfill_http_exact(
                    [
                        (
                            str(r.product_id),
                            r.gap_start.to_pydatetime(),
                            r.gap_end.to_pydatetime(),
                        )
                        for r in group.itertuples()
                    ],
                    granularity,
                    exchange=str(exchange),
                    protocol_label=f"HTTP {label}",
                )
        final_gaps = self._get_bag_gaps(deduped, window_start, window_end, granularity)
        return {
            "status": "perfected" if final_gaps.empty else "incomplete",
            "passes": max(max_passes, 1),
            "remaining_fragments": len(final_gaps),
        }

    def repair_http_overlap(
        self,
        pairs: List[str],
        granularity: str = "300",
        overlap_candles: int = 6,
        end: Optional[datetime] = None,
        exchange: str = "coinbase",
    ):
        if exchange != "coinbase" or not pairs:
            return
        repair_end = _floor_time_to_granularity(end or _utc_now_naive(), granularity)
        self.backfill_http_exact(
            [
                (
                    pid,
                    repair_end
                    - timedelta(seconds=int(granularity) * max(overlap_candles, 1)),
                    repair_end,
                )
                for pid in pairs
            ],
            granularity,
            exchange=exchange,
            protocol_label="HTTP repair/tail",
        )

    def ws_snapshot(
        self,
        pairs: List[str],
        granularity: str = "300",
        exchange: str = "coinbase",
        target_candles: int = 350,
        settle_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
    ):
        from coinbase.websocket import WSClient

        if exchange != "coinbase":
            return
        received, per_pair_timestamps, done, rows_all, lock, state = (
            set(),
            {pid: set() for pid in pairs},
            threading.Event(),
            [],
            threading.Lock(),
            {"last_new_data": time.monotonic()},
        )

        def on_message(msg):
            try:
                data = json.loads(msg) if isinstance(msg, str) else msg
            except Exception:
                return
            any_new = False
            for event in data.get("events", []):
                if event.get("type") != "snapshot":
                    continue
                for c in event.get("candles", []):
                    pid = c.get("product_id", "")
                    try:
                        ts = _parse_candle_timestamp(c["start"])
                        rows_all.append(
                            {
                                "exchange": exchange,
                                "product_id": pid,
                                "timestamp": ts,
                                "open": float(c["open"]),
                                "high": float(c["high"]),
                                "low": float(c["low"]),
                                "close": float(c["close"]),
                                "volume": float(c["volume"]),
                                "granularity": granularity,
                            }
                        )
                        with lock:
                            received.add(pid)
                            if (
                                pid in per_pair_timestamps
                                and ts not in per_pair_timestamps[pid]
                            ):
                                per_pair_timestamps[pid].add(ts)
                                any_new = True
                    except Exception:
                        pass
            if any_new:
                with lock:
                    state["last_new_data"] = time.monotonic()

        client = WSClient(api_key=None, api_secret=None, on_message=on_message)
        client.open()
        client.subscribe(pairs, ["candles"])
        deadline = time.monotonic() + max(timeout_seconds, 1.0)
        while time.monotonic() < deadline and not done.is_set():
            with lock:
                if (
                    per_pair_timestamps
                    and min(len(s) for s in per_pair_timestamps.values())
                    >= max(target_candles, 1)
                ) or (
                    received.issuperset(set(pairs))
                    and time.monotonic() - state["last_new_data"]
                    >= max(settle_seconds, 0.0)
                ):
                    done.set()
            if done.is_set():
                break
            time.sleep(0.25)
        client.close()
        if rows_all:
            self.save_candles(pd.DataFrame(rows_all))

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

        if exchange != "coinbase" or not pairs:
            return
        stop_event, pair_set, state = (
            stop_event or threading.Event(),
            set(pairs),
            {"last_activity": time.monotonic()},
        )

        def _decode_rows(msg):
            try:
                data = (
                    json.loads(msg)
                    if isinstance(msg, str)
                    else msg.to_dict()
                    if hasattr(msg, "to_dict")
                    else msg
                )
            except Exception:
                return pd.DataFrame()
            if not isinstance(data, dict):
                return pd.DataFrame()
            rows = []
            for event in data.get("events", []):
                for candle in event.get("candles", []):
                    if candle.get("product_id") not in pair_set:
                        continue
                    try:
                        ts = _parse_candle_timestamp(candle["start"])
                        if int(ts.timestamp()) % int(granularity) != 0:
                            continue
                        rows.append(
                            {
                                "exchange": exchange,
                                "product_id": candle["product_id"],
                                "timestamp": ts,
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

                def on_message(msg):
                    df = _decode_rows(msg)
                    if not df.empty:
                        self.save_candles(df)
                        state["last_activity"] = time.monotonic()
                        if on_rows:
                            try:
                                on_rows(df)
                            except Exception:
                                pass

                client = WSClient(api_key=None, api_secret=None, on_message=on_message)
                client.open()
                client.candles(pairs)
                while not stop_event.is_set():
                    if (
                        time.monotonic() - state["last_activity"]
                        >= idle_restart_seconds
                    ):
                        break
                    time.sleep(1.0)
            except Exception as exc:
                time.sleep(3.0)
            finally:
                if client:
                    try:
                        client.close()
                    except Exception:
                        pass

    def list_products(
        self, granularity: str = "300", exchange: Optional[str] = None
    ) -> List[str]:
        sql, params = (
            f"SELECT DISTINCT product_id FROM candles WHERE granularity = ?{' AND exchange = ?' if exchange else ''}",
            [granularity] + ([exchange] if exchange else []),
        )
        if _use_pool(self.db_path):
            return [r[0] for r in _pool().execute(sql, params)]
        with duckdb.connect(self.db_path) as conn:
            return [r[0] for r in conn.execute(sql, params).fetchall()]


# --- Coin Graph ---
@dataclass
class EdgeState:
    velocity: float = 0.0
    ptt: float = 0.0
    stop: float = 0.0
    hit_ptt: bool = False
    hit_stop: bool = False


@dataclass
class NodeState:
    height: float = 0.0


class CoinGraph:
    def __init__(self, fee_rate: float = 0.001, min_pair_coverage: float = 0.9):
        self.fee_rate, self.min_pair_coverage = fee_rate, min_pair_coverage
        self.nodes, self.all_pairs, self.bag_subscriptions = set(), [], []
        self.edges, self.edge_state, self.edge_product_id, self.edge_is_inverted = (
            {},
            {},
            {},
            {},
        )
        self.node_state, self.common_timestamps, self._volatility, self._vol_window = (
            {},
            [],
            defaultdict(list),
            20,
        )

    def add_product_frame(
        self,
        exchange: str,
        product_id: str,
        df: pd.DataFrame,
        *,
        coverage: Optional[float] = None,
    ) -> None:
        base, quote = product_id.split("-", 1)
        self.nodes.update([base, quote])
        self.node_state.setdefault(base, NodeState())
        self.node_state.setdefault(quote, NodeState())
        direct, reverse = (exchange, base, quote), (exchange, quote, base)
        self.edges[direct], self.edges[reverse] = df, df
        self.edge_state[direct], self.edge_state[reverse] = EdgeState(), EdgeState()
        self.edge_product_id[direct], self.edge_product_id[reverse] = (
            product_id,
            product_id,
        )
        self.edge_is_inverted[direct], self.edge_is_inverted[reverse] = False, True

    def edge_price_components(
        self, edge: Tuple[str, str, str], row
    ) -> Dict[str, float]:
        c = _finite_float(row.get("close", 0.0), 0.0)
        o = _finite_float(row.get("open", c), c)
        h = _finite_float(row.get("high", c), c)
        l = _finite_float(row.get("low", c), c)
        if not self.edge_is_inverted.get(edge, False):
            return {"open": o, "high": h, "low": l, "close": c}
        return {
            "open": 1.0 / o if o > 0 else 0.0,
            "high": 1.0 / l if l > 0 else 0.0,
            "low": 1.0 / h if h > 0 else 0.0,
            "close": 1.0 / c if c > 0 else 0.0,
        }

    def load(
        self,
        db_path: Optional[str] = None,
        granularity: str = None,
        min_partners: int = 5,
        max_partners: Optional[int] = None,
        lookback_days: int = 365,
        refresh_bag: bool = False,
        exchange: str = "coinbase",
        skip_fetch: bool = False,
        drawthrough_fetch: bool = False,
        use_cached_bag: bool = False,
        persist_bag: bool = False,
        min_pair_coverage: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        explicit_bag: Optional[List[Dict[str, str]]] = None,
    ) -> int:
        db_path, granularity = (
            db_path or str(Config.DB_PATH),
            granularity or Config.DEFAULT_GRANULARITY,
        )
        self.min_pair_coverage = (
            min_pair_coverage
            if min_pair_coverage is not None
            else self.min_pair_coverage
        )
        self.cache = CandleCache(db_path)
        self.cache.bootstrap_database(granularity=granularity, exchange=exchange)
        end = end_time if end_time is not None else _utc_now_naive()
        start = (
            start_time
            if start_time is not None
            else (end - timedelta(days=lookback_days))
        )

        bag_subscriptions = (
            explicit_bag
            if explicit_bag is not None
            else (
                _load_explicit_bag_subscriptions(str(Config.BAG_PATH))
                if use_cached_bag and not refresh_bag and Config.BAG_PATH.exists()
                else []
            )
        )
        if not bag_subscriptions:
            real_products = set(
                self.cache.list_products(granularity, exchange=exchange)
            )
            adj = {}
            for pid in real_products:
                b, q = pid.split("-", 1)
                adj.setdefault(b, set()).add(q)
                adj.setdefault(q, set()).add(b)
            coin_set = {
                c
                for c, p in adj.items()
                if len(p) >= min_partners
                and (max_partners is None or len(p) <= max_partners)
            }
            pairs = []
            seen = set()
            for pid in real_products:
                b, q = pid.split("-", 1)
                if (
                    q in {"GBP", "EUR", "SGD"}
                    or b in {"GBP", "EUR", "SGD"}
                    or (
                        q in ("USDC", "USDT")
                        and b
                        in {
                            p.split("-")[0] for p in real_products if p.endswith("-USD")
                        }
                    )
                ):
                    continue
                if (
                    b in coin_set
                    and q in coin_set
                    and tuple(sorted([b, q])) not in seen
                ):
                    seen.add(tuple(sorted([b, q])))
                    pairs.append(pid)
            bag_subscriptions = [
                {"exchange": exchange, "product_id": pid} for pid in pairs
            ]

        self.bag_subscriptions = bag_subscriptions
        if not skip_fetch:
            for ex, subs in defaultdict(
                list, {s["exchange"]: s for s in bag_subscriptions}
            ).items():
                if ex == "coinbase":
                    self.cache.repair_bag_drawthrough(
                        subs, start, end, granularity=granularity
                    )

        full_index = pd.date_range(
            start=start, end=end, freq=f"{granularity}s", inclusive="left"
        )
        values_sql = ", ".join(
            f"({_sql_escape(s['exchange'])}, {_sql_escape(s['product_id'])})"
            for s in _dedupe_subscriptions(bag_subscriptions)
        )
        sql = f"WITH bag(exchange, product_id) AS (VALUES {values_sql}) SELECT c.* FROM candles c JOIN bag b ON c.exchange = b.exchange AND c.product_id = b.product_id WHERE c.granularity = {_sql_escape(granularity)} AND c.timestamp >= {_sql_escape(start)} AND c.timestamp < {_sql_escape(end)} ORDER BY c.exchange, c.product_id, c.timestamp"
        if _use_pool(db_path):
            surface_df = pd.DataFrame(
                _pool().execute(sql),
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
                    "span",
                ],
            )
        else:
            with duckdb.connect(db_path) as conn:
                surface_df = conn.execute(sql).df()

        valid_pairs = []
        for (ex, pid), df in surface_df.groupby(["exchange", "product_id"], sort=False):
            df = (
                df.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .set_index("timestamp")
            )
            missing_idx = full_index.difference(df.index)
            df = df.reindex(full_index)
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["close"] = (
                df["close"]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            for col in ["open", "high", "low"]:
                df[col] = (
                    df[col]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(df["close"])
                    .fillna(0.0)
                )
            df["volume"] = df["volume"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if not df.empty:
                df.loc[missing_idx, "open"] = df.loc[missing_idx, "close"]
                df.loc[missing_idx, "high"] = df.loc[missing_idx, "close"]
                df.loc[missing_idx, "low"] = df.loc[missing_idx, "close"]
                df.loc[missing_idx, "volume"] = 0.0
            self.add_product_frame(ex, pid, df, coverage=1.0)
            valid_pairs.append(f"{ex}:{pid}")

        self.all_pairs = valid_pairs
        # Preferentially surface prime fiat nodes so routing and wallet
        # initialization can pick them as value assets (e.g. PBUSD).
        PRIME_ORDER = ("PBUSD", "USD", "USDT", "USDC", "BUSD")
        found_primes = [p for p in PRIME_ORDER if p in self.nodes]
        if found_primes:
            for p in found_primes:
                self.nodes.discard(p)
            self.nodes = set(found_primes) | self.nodes
        self._align_timestamps()
        return len(self.common_timestamps)

    def _align_timestamps(self):
        if not self.edges:
            return
        all_indices = [
            set(df.index)
            for i, df in enumerate(self.edges.values())
            if id(df) not in {id(d) for d in list(self.edges.values())[:i]}
        ]
        if not all_indices:
            return
        common = all_indices[0]
        for idx in all_indices[1:]:
            common = common.intersection(idx)
        self.common_timestamps = sorted(list(common))

    def update(
        self, bar_idx: int
    ) -> Tuple[
        Dict[Tuple[str, str, str], float],
        Dict[Tuple[str, str, str], float],
        Dict[Tuple[str, str, str], bool],
        Dict[Tuple[str, str, str], bool],
    ]:
        if bar_idx >= len(self.common_timestamps):
            return {}, {}, {}, {}
        ts = self.common_timestamps[bar_idx]
        edge_accels, edge_velocities, hit_ptt, hit_stop = {}, {}, {}, {}
        for edge, df in self.edges.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            prices = self.edge_price_components(edge, row)
            velocity = (
                np.log(prices["close"] / prices["open"]) if prices["open"] > 0 else 0.0
            )
            accel = velocity - self.edge_state[edge].velocity
            self.edge_state[edge].velocity = velocity
            edge_accels[edge], edge_velocities[edge] = accel, velocity
            self._volatility[edge].append(abs(velocity))
            if len(self._volatility[edge]) > self._vol_window:
                self._volatility[edge].pop(0)
            vol = np.mean(self._volatility[edge]) if self._volatility[edge] else 0.0
            self.edge_state[edge].ptt, self.edge_state[edge].stop = (
                self.fee_rate + vol,
                -(self.fee_rate + vol),
            )
            hit_ptt[edge], hit_stop[edge] = (
                velocity > self.edge_state[edge].ptt,
                velocity < self.edge_state[edge].stop,
            )
            self.edge_state[edge].hit_ptt, self.edge_state[edge].hit_stop = (
                hit_ptt[edge],
                hit_stop[edge],
            )
        outflow = defaultdict(list)
        for (_, base, _), accel in edge_accels.items():
            outflow[base].append(accel)
        for node in self.node_state:
            self.node_state[node].height = (
                np.mean(outflow[node]) if outflow[node] else 0.0
            )
        return edge_accels, edge_velocities, hit_ptt, hit_stop
