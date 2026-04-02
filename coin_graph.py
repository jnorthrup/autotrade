import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import Config
from candle_cache import CandleCache, _utc_now_naive
from pool_client import PoolClient, pool_is_running, DEFAULT_SOCKET

DEFAULT_MIN_PAIR_COVERAGE = 0.9


def _use_pool() -> bool:
    """True if literbike pool server is running and responsive."""
    try:
        return pool_is_running()
    except Exception:
        return False


def _pool() -> PoolClient:
    """Get a PoolClient connected to the shared server."""
    return PoolClient(DEFAULT_SOCKET)


def _adjacency_from_products(product_ids: List[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    real_products: Set[str] = set()
    adjacency: Dict[str, Set[str]] = {}
    for pid in product_ids:
        parts = pid.split("-", 1)
        if len(parts) != 2:
            continue
        base, quote = parts
        real_products.add(pid)
        adjacency.setdefault(base, set()).add(quote)
        adjacency.setdefault(quote, set()).add(base)
    return real_products, adjacency


def _normalize_bag_subscription(entry, default_exchange: Optional[str] = None) -> Optional[Dict[str, str]]:
    if isinstance(entry, dict):
        exchange = str(entry.get("exchange") or default_exchange or "").strip()
        product_id = entry.get("product_id") or entry.get("pair") or entry.get("symbol")
    elif isinstance(entry, str):
        raw = entry.strip()
        if not raw:
            return None
        if ":" in raw:
            maybe_exchange, maybe_product = raw.split(":", 1)
            if maybe_exchange and "-" in maybe_product:
                exchange = maybe_exchange.strip()
                product_id = maybe_product.strip()
            else:
                exchange = str(default_exchange or "").strip()
                product_id = raw
        else:
            exchange = str(default_exchange or "").strip()
            product_id = raw
    else:
        return None

    product_id = str(product_id or "").strip()
    if not exchange or "-" not in product_id:
        return None
    return {"exchange": exchange, "product_id": product_id}


def _normalize_bag_subscriptions(entries, default_exchange: Optional[str] = None) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for entry in entries or []:
        sub = _normalize_bag_subscription(entry, default_exchange=default_exchange)
        if sub is not None:
            normalized.append(sub)
    return normalized


def _load_explicit_bag_subscriptions(bag_path: str) -> List[Dict[str, str]]:
    with open(bag_path, "r") as f:
        raw_entries = json.load(f)
    if not isinstance(raw_entries, list):
        raise ValueError(f"Bag file must contain a JSON list of subscriptions: {bag_path}")

    deduped: Dict[Tuple[str, str], Dict[str, str]] = {}
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Bag file entry {idx} must be an object with exchange and product_id: {entry!r}"
            )
        exchange = str(entry.get("exchange") or "").strip()
        product_id = str(entry.get("product_id") or "").strip()
        if not exchange or not product_id or "-" not in product_id:
            raise ValueError(
                f"Bag file entry {idx} must include explicit exchange and product_id: {entry!r}"
            )
        sub = {"exchange": exchange, "product_id": product_id}
        deduped[(exchange, product_id)] = sub
    return list(deduped.values())


@dataclass
class EdgeState:
    velocity: float = 0.0
    ptt: float = 0.0      # upper band = profit target
    stop: float = 0.0     # lower band = stop loss
    hit_ptt: bool = False
    hit_stop: bool = False


@dataclass  
class NodeState:
    height: float = 0.0


class CoinGraph:
    def __init__(self, fee_rate: float = 0.001, min_pair_coverage: float = DEFAULT_MIN_PAIR_COVERAGE):
        self.fee_rate = fee_rate
        self.min_pair_coverage = min_pair_coverage
        self.nodes: Set[str] = set()
        self.all_pairs: List[str] = []
        self.bag_subscriptions: List[Dict[str, str]] = []
        self.edges: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        self.edge_state: Dict[Tuple[str, str, str], EdgeState] = {}
        self.edge_product_id: Dict[Tuple[str, str, str], str] = {}
        self.edge_is_inverted: Dict[Tuple[str, str, str], bool] = {}
        self.node_state: Dict[str, NodeState] = {}
        self.common_timestamps: List[pd.Timestamp] = []
        self.pair_coverage: Dict[str, float] = {}
        self.pair_exchange: Dict[str, str] = {}
        self.bag_id: Optional[str] = None
        self.bag_window_id: Optional[str] = None
        self.bag_surface_name: Optional[str] = None
        self.bag_thresholds_view_name: Optional[str] = None
        self._volatility: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        self._vol_window = 20

    def add_product_frame(
        self,
        exchange: str,
        product_id: str,
        df: pd.DataFrame,
        *,
        coverage: Optional[float] = None,
    ) -> None:
        parts = product_id.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid product_id for edge registration: {product_id!r}")

        base, quote = parts
        self.nodes.add(base)
        self.nodes.add(quote)
        self.node_state.setdefault(base, NodeState())
        self.node_state.setdefault(quote, NodeState())

        direct_edge = (exchange, base, quote)
        reverse_edge = (exchange, quote, base)
        self.edges[direct_edge] = df
        self.edges[reverse_edge] = df
        self.edge_state[direct_edge] = EdgeState()
        self.edge_state[reverse_edge] = EdgeState()
        self.edge_product_id[direct_edge] = product_id
        self.edge_product_id[reverse_edge] = product_id
        self.edge_is_inverted[direct_edge] = False
        self.edge_is_inverted[reverse_edge] = True

        if coverage is not None:
            subscription_key = f"{exchange}:{product_id}"
            self.pair_coverage[subscription_key] = coverage
            self.pair_exchange[subscription_key] = exchange

    def edge_price_components(self, edge: Tuple[str, str, str], row) -> Dict[str, float]:
        close = float(row.get("close", 0.0) or 0.0)
        open_price = float(row.get("open", close) or close or 0.0)
        high = float(row.get("high", close) or close or 0.0)
        low = float(row.get("low", close) or close or 0.0)

        if not self.edge_is_inverted.get(edge, False):
            return {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }

        inv_open = 1.0 / open_price if open_price > 0 else 0.0
        inv_close = 1.0 / close if close > 0 else 0.0
        inv_high = 1.0 / low if low > 0 else 0.0
        inv_low = 1.0 / high if high > 0 else 0.0
        return {
            "open": inv_open,
            "high": inv_high,
            "low": inv_low,
            "close": inv_close,
        }

    def load(self, db_path: Optional[str] = None, granularity: str = None,
             min_partners: int = 5, max_partners: Optional[int] = None,
             lookback_days: int = 365, refresh_bag: bool = False,
             exchange: str = "coinbase", skip_fetch: bool = False,
             drawthrough_fetch: bool = False, use_cached_bag: bool = False,
             persist_bag: bool = False, min_pair_coverage: Optional[float] = None) -> int:
        db_path = db_path or str(Config.DB_PATH)
        granularity = granularity or Config.DEFAULT_GRANULARITY
        min_pair_coverage = (
            self.min_pair_coverage if min_pair_coverage is None else min_pair_coverage
        )

        self.cache = CandleCache(db_path)
        bootstrap = self.cache.bootstrap_database(
            granularity=granularity,
            exchange=exchange,
        )
        if bootstrap["normalized_timestamps"] or bootstrap["purged_future_rows"]:
            print(
                f"[CoinGraph bootstrap] normalized={bootstrap['normalized_timestamps']} "
                f"purged={bootstrap['purged_future_rows']} "
                f"exchange={exchange} granularity={granularity}"
            )
        end = _utc_now_naive()
        start = end - timedelta(days=lookback_days)

        bag_subscriptions: List[Dict[str, str]] = []
        if use_cached_bag and not refresh_bag and Config.BAG_PATH.exists():
            bag_subscriptions = _load_explicit_bag_subscriptions(str(Config.BAG_PATH))
            print(f"Loaded bag of {len(bag_subscriptions)} subscriptions from {Config.BAG_PATH}")

        if not bag_subscriptions:
            historical_pairs = self.cache.historical_products(
                start,
                end,
                granularity=granularity,
                exchange=exchange,
                min_coverage_ratio=min_pair_coverage,
            )
            pairs: List[str] = []
            if historical_pairs:
                real_products, adjacency = _adjacency_from_products(historical_pairs)
                print(
                    f"[CoinGraph] using historical universe from DuckDB "
                    f"({len(real_products)} products, coverage>={min_pair_coverage:.2f})"
                )
            elif exchange == "binance":
                real_products = set()
                adjacency = {}
                try:
                    if _use_pool():
                        rows = _pool().execute("SELECT DISTINCT product_id FROM candles WHERE exchange = ?", [exchange])
                        real_products, adjacency = _adjacency_from_products([r[0] for r in rows])
                    else:
                        import duckdb
                        with duckdb.connect(db_path) as conn:
                            rows = conn.execute(
                                "SELECT DISTINCT product_id FROM candles WHERE exchange = ?",
                                [exchange],
                            ).fetchall()
                            real_products, adjacency = _adjacency_from_products([r[0] for r in rows])
                except Exception as e:
                    print(f"Error reading products from DuckDB: {e}")
                    real_products = set()
                    adjacency = {}
            else:
                resp = self.cache.client.get_public_products()
                adjacency = {}
                real_products = set()
                for p in resp.products:
                    if p.status != "online" or p.trading_disabled:
                        continue
                    pid = p.product_id
                    real_products.add(pid)
                real_products, adjacency = _adjacency_from_products(sorted(real_products))

            coin_set = {
                c for c, partners in adjacency.items() 
                if len(partners) >= min_partners and (max_partners is None or len(partners) <= max_partners)
            }

            FIAT_EXCLUDE = {"GBP", "EUR", "SGD"}
            if exchange == "binance":
                FIAT_EXCLUDE = {"GBP", "EUR", "SGD"}
            usd_bases = {p.split("-")[0] for p in real_products if p.endswith("-USD")}

            seen = set()
            for pid in real_products:
                base, quote = pid.split("-", 1)
                if quote in FIAT_EXCLUDE or base in FIAT_EXCLUDE:
                    continue
                if quote in ("USDC", "USDT") and base in usd_bases:
                    continue
                if base in coin_set and quote in coin_set:
                    canonical = tuple(sorted([base, quote]))
                    if canonical not in seen:
                        seen.add(canonical)
                        pairs.append(pid)

            if not pairs and historical_pairs:
                print(
                    "[CoinGraph] historical universe did not satisfy partner filter; "
                    "relaxing to available historical pairs"
                )
                seen.clear()
                for pid in real_products:
                    base, quote = pid.split("-", 1)
                    if quote in FIAT_EXCLUDE or base in FIAT_EXCLUDE:
                        continue
                    if quote in ("USDC", "USDT") and base in usd_bases:
                        continue
                    canonical = tuple(sorted([base, quote]))
                    if canonical not in seen:
                        seen.add(canonical)
                        pairs.append(pid)
            
            if persist_bag:
                try:
                    import json
                    with open(Config.BAG_PATH, 'w') as f:
                        json.dump(
                            [{"exchange": exchange, "product_id": pid} for pid in pairs],
                            f,
                            indent=4,
                        )
                    print(f"Saved bag of {len(pairs)} subscriptions to {Config.BAG_PATH}")
                except Exception as e:
                    print(f"Error saving bag: {e}")
            bag_subscriptions = [{"exchange": exchange, "product_id": pid} for pid in pairs]

        self.bag_subscriptions = bag_subscriptions
        self.pair_exchange = {}

        pairs_by_exchange: Dict[str, List[str]] = defaultdict(list)
        for sub in bag_subscriptions:
            pairs_by_exchange[sub["exchange"]].append(sub["product_id"])

        total_pairs = sum(len(v) for v in pairs_by_exchange.values())
        print(f"Graph discovery: {total_pairs} pairs across {len(pairs_by_exchange)} exchanges")

        if not skip_fetch:
            for bag_exchange, bag_pairs in pairs_by_exchange.items():
                cached_pairs = set(
                    self.cache.cached_products_in_range(
                        bag_pairs,
                        start,
                        end,
                        granularity,
                        exchange=bag_exchange,
                    )
                )
                missing_pairs = [pid for pid in bag_pairs if pid not in cached_pairs]

                if bag_exchange == "binance":
                    if missing_pairs or bag_pairs:
                        print("[CoinGraph] Binance mode: importing archive CSVs into DuckDB")
                        self.cache.import_binance_archive(missing_pairs or bag_pairs, granularity)
                    continue

                if not missing_pairs:
                    print(f"[CoinGraph] {bag_exchange}: local DuckDB already covers requested range; skipping fetch")
                    continue
                if Config.USE_WS_ONLY:
                    print(f"[CoinGraph] {bag_exchange}: USE_WS_ONLY: ws_snapshot")
                    try:
                        self.cache.ws_snapshot(missing_pairs, granularity, exchange=bag_exchange)
                    except Exception as e:
                        print(f"[CoinGraph] WS snapshot failed: {e}")
                elif lookback_days <= 2:
                    self.cache.ws_snapshot(missing_pairs, granularity, exchange=bag_exchange)
                elif drawthrough_fetch:
                    print(
                        f"[CoinGraph] {bag_exchange}: draw-through fetch: seeding snapshot and backfilling "
                        f"{len(missing_pairs)} missing pairs in background"
                    )
                    try:
                        self.cache.ws_snapshot(missing_pairs, granularity, exchange=bag_exchange)
                    except Exception as e:
                        print(f"[CoinGraph] WS snapshot failed: {e}")
                    self.cache.prefetch_all_async(
                        missing_pairs,
                        start,
                        end,
                        granularity,
                        name=f"drawthrough-{bag_exchange}-{granularity}",
                        exchange=bag_exchange,
                    )
                else:
                    try:
                        self.cache.ws_snapshot(missing_pairs, granularity, exchange=bag_exchange)
                    except Exception as e:
                        print(f"[CoinGraph] WS snapshot failed: {e}")
                    self.cache.prefetch_all(missing_pairs, start, end, granularity, exchange=bag_exchange)

        statuses = self.cache.verify_bag_contiguous_coverage(
            bag_subscriptions,
            start,
            end,
            granularity=granularity,
        )
        status_by_key = {
            (str(status["exchange"]), str(status["product_id"])): status
            for status in statuses
        }
        valid_subscriptions: List[Dict[str, str]] = []
        for sub in bag_subscriptions:
            status = status_by_key.get((sub["exchange"], sub["product_id"]))
            if status is None or not bool(status["covered"]):
                coverage = 0.0 if status is None else float(status["coverage_ratio"])
                print(
                    f"[CoinGraph] dropping {sub['exchange']}:{sub['product_id']}: "
                    f"coverage={coverage:.3f} below contiguous minimum {min_pair_coverage:.3f}"
                )
                continue
            valid_subscriptions.append(sub)

        self.bag_id = None
        self.bag_window_id = None
        self.bag_thresholds_view_name = None
        self.bag_surface_name = None
        if statuses:
            bag_state = self.cache.persist_bag_window_status(
                bag_subscriptions,
                statuses,
                start,
                end,
                granularity=granularity,
                min_coverage_ratio=min_pair_coverage,
                bag_name="coin_graph",
            )
            self.bag_id = bag_state["bag_id"]
            self.bag_window_id = bag_state["bag_window_id"]
            self.bag_thresholds_view_name = "bag_thresholds_v"
        if valid_subscriptions and self.bag_window_id is not None:
            self.bag_surface_name = self.cache.materialize_bag_surface_for_window(
                self.bag_window_id
            )
            surface_df = self.cache.read_bag_surface(self.bag_surface_name)
        else:
            surface_df = pd.DataFrame()

        valid_pairs: List[str] = []
        if not surface_df.empty:
            grouped = surface_df.groupby(["exchange", "product_id"], sort=False)
        else:
            grouped = []

        for (exchange, product_id), df in grouped:
            coverage = float(status_by_key[(exchange, product_id)]["coverage_ratio"])
            df = (
                df.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .set_index("timestamp")
            )
            self.add_product_frame(exchange, product_id, df, coverage=coverage)
            valid_pairs.append(f"{exchange}:{product_id}")

        self.all_pairs = valid_pairs
        self.bag_subscriptions = [{"exchange": self.pair_exchange[pid], "product_id": pid} for pid in valid_pairs]
        
        if "USD" in self.nodes:
            self.nodes.discard("USD")
            self.nodes = {"USD"} | self.nodes

        self._align_timestamps()
        return len(self.common_timestamps)

    def _align_timestamps(self):
        if not self.edges:
            return
        all_indices = []
        seen_frames = set()
        for df in self.edges.values():
            frame_id = id(df)
            if frame_id in seen_frames:
                continue
            seen_frames.add(frame_id)
            all_indices.append(set(df.index))
        if not all_indices:
            return
        common = all_indices[0]
        for idx in all_indices[1:]:
            common = common.intersection(idx)
        self.common_timestamps = sorted(list(common))
        print(
            f"Aligned {len(self.common_timestamps)} common bars across "
            f"{len(all_indices)} canonical pairs and {len(self.nodes)} nodes"
        )

    def update(self, bar_idx: int) -> Tuple[Dict[Tuple[str, str, str], float], Dict[Tuple[str, str, str], float], 
                                            Dict[Tuple[str, str, str], bool], Dict[Tuple[str, str, str], bool]]:
        """Compute velocity and PTT/STOP band crossings.
        
        Returns: (edge_accels, edge_velocities, hit_ptt, hit_stop)
        """
        if bar_idx >= len(self.common_timestamps):
            return {}, {}, {}, {}

        ts = self.common_timestamps[bar_idx]
        edge_accels: Dict[Tuple[str, str, str], float] = {}
        edge_velocities: Dict[Tuple[str, str, str], float] = {}
        hit_ptt: Dict[Tuple[str, str, str], bool] = {}
        hit_stop: Dict[Tuple[str, str, str], bool] = {}

        for edge, df in self.edges.items():
            if ts not in df.index:
                continue

            row = df.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            prices = self.edge_price_components(edge, row)
            close = prices["close"]
            open_price = prices["open"]

            velocity = 0.0
            if open_price > 0:
                velocity = np.log(close / open_price)

            prev_velocity = self.edge_state[edge].velocity
            accel = velocity - prev_velocity
            
            self.edge_state[edge].velocity = velocity
            edge_accels[edge] = accel
            edge_velocities[edge] = velocity
            
            # Track volatility for band computation
            self._volatility[edge].append(abs(velocity))
            if len(self._volatility[edge]) > self._vol_window:
                self._volatility[edge].pop(0)
            
            # Compute bands: fee + volatility noise floor
            vol = np.mean(self._volatility[edge]) if self._volatility[edge] else 0.0
            self.edge_state[edge].ptt = self.fee_rate + vol
            self.edge_state[edge].stop = -(self.fee_rate + vol)
            
            # Check band crossings
            hit_ptt[edge] = velocity > self.edge_state[edge].ptt
            hit_stop[edge] = velocity < self.edge_state[edge].stop
            self.edge_state[edge].hit_ptt = hit_ptt[edge]
            self.edge_state[edge].hit_stop = hit_stop[edge]

        self._compute_heights(edge_accels)
        return edge_accels, edge_velocities, hit_ptt, hit_stop

    def _compute_heights(self, edge_accels: Dict[Tuple[str, str, str], float]):
        """Node height = mean outgoing accel."""
        outflow = defaultdict(list)
        for (_, base, _), accel in edge_accels.items():
            outflow[base].append(accel)
        
        for node in self.node_state:
            self.node_state[node].height = np.mean(outflow[node]) if outflow[node] else 0.0


if __name__ == "__main__":
    g = CoinGraph()
    n_bars = g.load()
    print(f"Loaded {len(g.nodes)} nodes, {len(g.edges)} edges, {n_bars} bars")
