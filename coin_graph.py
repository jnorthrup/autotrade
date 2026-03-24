from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import Config
from candle_cache import CandleCache


@dataclass
class EdgeState:
    velocity: float = 0.0


@dataclass  
class NodeState:
    height: float = 0.0


class CoinGraph:
    def __init__(self, fee_rate: float = 0.001):
        self.fee_rate = fee_rate
        self.nodes: Set[str] = set()
        self.all_pairs: List[str] = []
        self.edges: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.edge_state: Dict[Tuple[str, str], EdgeState] = {}
        self.node_state: Dict[str, NodeState] = {}
        self.common_timestamps: List[pd.Timestamp] = []

    def load(self, db_path: Optional[str] = None, granularity: str = None,
             min_partners: int = 5, max_partners: Optional[int] = None,
             lookback_days: int = 365, refresh_bag: bool = False,
             exchange: str = "coinbase", skip_fetch: bool = False) -> int:
        db_path = db_path or str(Config.DB_PATH)
        granularity = granularity or Config.DEFAULT_GRANULARITY

        self.cache = CandleCache(db_path)

        pairs = []
        if not refresh_bag and Config.BAG_PATH.exists():
            try:
                import json
                with open(Config.BAG_PATH, 'r') as f:
                    pairs = json.load(f)
                print(f"Loaded bag of {len(pairs)} pairs from {Config.BAG_PATH}")
            except Exception as e:
                print(f"Error loading bag: {e}")
                pairs = []

        if not pairs:
            resp = self.cache.client.get_public_products()
            adjacency = {}
            real_products = set()
            for p in resp.products:
                if p.status != "online" or p.trading_disabled:
                    continue
                parts = p.product_id.split("-", 1)
                if len(parts) != 2:
                    continue
                base, quote = parts
                real_products.add(p.product_id)
                adjacency.setdefault(base, set()).add(quote)
                adjacency.setdefault(quote, set()).add(base)

            coin_set = {
                c for c, partners in adjacency.items() 
                if len(partners) >= min_partners and (max_partners is None or len(partners) <= max_partners)
            }

            FIAT_EXCLUDE = {"GBP", "EUR"}
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
            
            try:
                import json
                with open(Config.BAG_PATH, 'w') as f:
                    json.dump(pairs, f, indent=4)
                print(f"Saved bag of {len(pairs)} pairs to {Config.BAG_PATH}")
            except Exception as e:
                print(f"Error saving bag: {e}")

        self.all_pairs = pairs
        print(f"Graph discovery: {len(pairs)} pairs")

        end = datetime.now()
        start = end - timedelta(days=lookback_days)

        if skip_fetch:
            pass
        elif exchange == "binance":
            print("[CoinGraph] Binance mode: importing archive CSVs into DuckDB")
            pairs = self.cache.import_binance_archive(pairs, granularity)
        elif Config.USE_WS_ONLY:
            print("[CoinGraph] USE_WS_ONLY: ws_snapshot")
            try:
                self.cache.ws_snapshot(pairs, granularity)
            except Exception as e:
                print(f"[CoinGraph] WS snapshot failed: {e}")
        elif lookback_days <= 2:
            self.cache.ws_snapshot(pairs, granularity)
        else:
            try:
                self.cache.ws_snapshot(pairs, granularity)
            except Exception as e:
                print(f"[CoinGraph] WS snapshot failed: {e}")
            self.cache.prefetch_all(pairs, start, end, granularity)

        for product_id in pairs:
            base, quote = product_id.split("-", 1)
            df = self.cache.get_candles(product_id, start, end, granularity)
            if df.empty:
                continue
            df = df.set_index('timestamp')
            self.nodes.add(base)
            self.nodes.add(quote)
        
        # Ensure USD is countercoin-0 (ground truth for PnL scoring)
        if "USD" in self.nodes:
            self.nodes.discard("USD")
            self.nodes = {"USD"} | self.nodes
            self.edges[(base, quote)] = df
            self.edges[(quote, base)] = df
            self.edge_state[(base, quote)] = EdgeState()
            self.edge_state[(quote, base)] = EdgeState()
            self.node_state.setdefault(base, NodeState())
            self.node_state.setdefault(quote, NodeState())

        self._align_timestamps()
        return len(self.common_timestamps)

    def _align_timestamps(self):
        if not self.edges:
            return
        all_indices = [df.index for df in self.edges.values()]
        common = set(all_indices[0])
        for idx in all_indices[1:]:
            common = common.union(idx)
        self.common_timestamps = sorted(list(common))
        print(f"Aligned {len(self.common_timestamps)} bars across {len(self.nodes)} nodes")

    def update(self, bar_idx: int) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
        """Compute velocity (log return) for each edge at bar_idx."""
        if bar_idx >= len(self.common_timestamps):
            return {}, {}

        ts = self.common_timestamps[bar_idx]
        edge_accels = {}
        edge_velocities = {}

        for (base, quote), df in self.edges.items():
            if ts not in df.index:
                continue

            row = df.loc[ts]
            close = row.get('close', 0)
            open_price = row.get('open', close)

            velocity = 0.0
            if open_price > 0:
                velocity = np.log(close / open_price)

            prev_velocity = self.edge_state[(base, quote)].velocity
            accel = velocity - prev_velocity
            
            self.edge_state[(base, quote)].velocity = velocity
            edge_accels[(base, quote)] = accel
            edge_velocities[(base, quote)] = velocity

        self._compute_heights(edge_accels)
        return edge_accels, edge_velocities

    def _compute_heights(self, edge_accels: Dict[Tuple[str, str], float]):
        """Node height = mean outgoing accel."""
        outflow = defaultdict(list)
        for (base, quote), accel in edge_accels.items():
            outflow[base].append(accel)
        
        for node in self.node_state:
            self.node_state[node].height = np.mean(outflow[node]) if outflow[node] else 0.0


if __name__ == "__main__":
    g = CoinGraph()
    n_bars = g.load()
    print(f"Loaded {len(g.nodes)} nodes, {len(g.edges)} edges, {n_bars} bars")
