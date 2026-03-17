import heapq
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import Config
from candle_cache import CandleCache


@dataclass
class EdgeState:
    base: str
    quote: str
    conductance: float = 1.0
    reverse_conductance: float = 1.0
    velocity: float = 0.0
    reverse_velocity: float = 0.0
    accel: float = 0.0
    reverse_accel: float = 0.0
    cumulative_pnl: float = 0.0
    volume: float = 0.0
    n_traversals: int = 0
    last_pnl: float = 0.0
    short_term_memory: List[float] = field(default_factory=list)
    long_term_memory: List[float] = field(default_factory=list)
    streak: int = 0
    temperature: float = 1.0
    volatility_window: List[float] = field(default_factory=list)
    wins: int = 0
    losses: int = 0


@dataclass
class NodeState:
    currency: str
    height: float = 0.0
    net_inflow: float = 0.0
    time_at_north: int = 0
    total_bars: int = 0


class CoinGraph:
    def __init__(self, fee_rate: float = 0.001):
        self.fee_rate = fee_rate
        self.nodes: Set[str] = set()
        self.edges: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.edge_state: Dict[Tuple[str, str], EdgeState] = {}
        self.node_state: Dict[str, NodeState] = {}
        self.common_timestamps: List[pd.Timestamp] = []
        self._accel_history: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._history_window = 32

    def load(self, db_path: Optional[str] = None, granularity: str = None) -> int:
        db_path = db_path or str(Config.DB_PATH)
        granularity = granularity or Config.DEFAULT_GRANULARITY
        
        self.cache = CandleCache(db_path)
        
        import duckdb
        if not Path(db_path).exists():
            print(f"Database {db_path} not found.")
            return 0
            
        self.cache = CandleCache(db_path)
        
        products = self.cache.list_products(granularity)
            
        for product_id in products:
            if "-" not in product_id:
                continue
                
            base, quote = product_id.split("-", 1)
            self.nodes.add(base)
            self.nodes.add(quote)
            
            # FAST LOAD: Just query what the DB has right now
            df = self.cache.query(f"SELECT * FROM candles WHERE product_id = ? AND granularity = ? ORDER BY timestamp", 
                                 [product_id, granularity])
            
            if df.empty:
                continue
                
            df = df.set_index('timestamp')
            self.edges[(base, quote)] = df
            # Explicitly register the reverse edge so Dijkstra can construct full paths natively
            self.edges[(quote, base)] = df
            
            self.edge_state[(base, quote)] = EdgeState(base=base, quote=quote)
            self.edge_state[(quote, base)] = EdgeState(base=base, quote=quote)  # original direction — triggers negation in update()
            
            self.node_state.setdefault(base, NodeState(currency=base))
            self.node_state.setdefault(quote, NodeState(currency=quote))
        
        self._align_timestamps()
        return len(self.common_timestamps)

    def hydrate_increment(self, days: int = 7) -> int:
        """
        Actively pulls more data from the cache (potentially API) for all pairs.
        Used by the simulation loop to 'keep proceeding' when cache is partial.
        """
        granularity = Config.DEFAULT_GRANULARITY
        
        # Calculate start point based on existing data or 1 year back
        # If we have no data, start 1 year back. 
        # If we have data, start from the latest common timestamp.
        if self.common_timestamps:
            start_ts = self.common_timestamps[-1]
        else:
            start_ts = datetime.now() - timedelta(days=365)
            
        end_ts = start_ts + timedelta(days=days)
        if end_ts > datetime.now():
            end_ts = datetime.now()
            
        if start_ts >= end_ts:
            return 0
            
        new_candles_count = 0
        print(f"Hydrating graph from {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')}...")
        
        for (base, quote) in list(self.edges.keys()):
            product_id = f"{base}-{quote}"
            # This triggers the draw-through
            df = self.cache.get_candles(product_id, start_ts, end_ts, granularity)
            
            if not df.empty:
                # Merge into existing edge data
                df = df.set_index('timestamp')
                existing_df = self.edges.get((base, quote), pd.DataFrame())
                
                if not existing_df.empty:
                    # Combined and deduplicate
                    combined = pd.concat([existing_df, df])
                    # Handle indices that might be overlapping
                    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                    self.edges[(base, quote)] = combined
                    self.edges[(quote, base)] = combined
                else:
                    self.edges[(base, quote)] = df
                    self.edges[(quote, base)] = df
                
                new_candles_count += len(df)
        
        # Re-align everything to incorporate new timestamps
        old_count = len(self.common_timestamps)
        self._align_timestamps()
        new_count = len(self.common_timestamps)
        
        return new_count - old_count

    def _align_timestamps(self):
        if not self.edges:
            return
            
        all_indices = [df.index for df in self.edges.values()]
        common = set(all_indices[0])
        for idx in all_indices[1:]:
            common = common.union(idx)
        self.common_timestamps = sorted(list(common))
        print(f"Aligned to {len(self.common_timestamps)} total 5m bars across {len(self.nodes)} nodes")

    def update(self, bar_idx: int) -> Dict[str, float]:
        if bar_idx >= len(self.common_timestamps):
            return {}
            
        ts = self.common_timestamps[bar_idx]
        
        edge_accels = {}
        
        for (base, quote), df in self.edges.items():
            if ts not in df.index:
                continue
                
            row = df.loc[ts]
            close = row.get('close', 0)
            open_price = row.get('open', close)
            
            log_return = 0.0
            if open_price > 0:
                # If we are looking at the reverse pair (quote, base) then the price action implies reciprocal return
                if (base, quote) in self.edges and hasattr(df, 'name') == False:
                    # Need a way to tell if this is the original or inverse edge
                    # We can check if quote is the *actual* quote of the product_id by checking the df origin, 
                    # but simpler: compute normal log return, then inverse it if it's the reverse edge
                    pass
                
                log_return = np.log(close / open_price)
                
                # Check if this edge is the inverted one (i.e. 'USD', 'BTC' instead of 'BTC', 'USD')
                # We can determine this by checking if the edge tuple format matches the standard convention
                # For now, let's explicitly store the 'original_direction' in EdgeState
                
            edge_state = self.edge_state[(base, quote)]
            
            # Inverse log_return if this is a backward edge. 
            if edge_state.base != base:
                log_return = -log_return
                
            accel = log_return - edge_state.velocity
            edge_state.velocity = log_return
            edge_state.accel = accel
            edge_state.volume = row.get('volume', 0.0)
            
            self._accel_history[(base, quote)].append(accel)
            if len(self._accel_history[(base, quote)]) > self._history_window:
                self._accel_history[(base, quote)].pop(0)
            
            edge_accels[(base, quote)] = accel
        
        self._compute_heights(edge_accels)
        
        for ns in self.node_state.values():
            ns.total_bars += 1
        
        return edge_accels

    def _compute_heights(self, edge_accels: Dict[Tuple[str, str], float]):
        outflow_accels = defaultdict(list)
        inflow_accels = defaultdict(list)
        
        for (base, quote), accel in edge_accels.items():
            outflow_accels[base].append(accel)
            inflow_accels[quote].append(accel)
        
        for node, ns in self.node_state.items():
            # Because we explicitly modeled forward AND backward edges with inverted accelerations,
            # we don't need to subtract inflow from outflow anymore. We can just sum them natively,
            # or simply look at the absolute peak outgoing acceleration potential to see if it is
            # "pulling" capital towards itself from other nodes.
            out_mean = np.mean(outflow_accels[node]) if outflow_accels[node] else 0.0
            
            ns.height = out_mean
            
            if ns.height > 0:
                ns.time_at_north += 1

    def edge_weight(self, base: str, quote: str) -> float:
        edge = (base, quote)
        if edge not in self.edge_state:
            return float('inf')
        
        es = self.edge_state[edge]
        
        # We don't artificially clip the model's prediction here anymore as it prevents routing.
        # The PyTorch model prediction naturally provides the scale, and the conductance scales it.
        w = self.fee_rate - es.accel * es.conductance
        return max(0.0001, w)

    def dijkstra(self, source: str) -> Dict[str, Tuple[float, List[str]]]:
        if source not in self.nodes:
            return {}
        
        dist = {source: 0.0}
        prev = {source: None}
        heap = [(0.0, source)]
        visited = set()
        
        while heap:
            curr_dist, u = heapq.heappop(heap)
            
            if u in visited:
                continue
            visited.add(u)
            
            for v in self.nodes:
                if u == v:
                    continue
                
                w = self.edge_weight(u, v)
                if w == float('inf'):
                    continue
                
                alt = curr_dist + w
                if v not in dist or alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))
        
        result = {}
        for target in self.nodes:
            if target == source or target not in prev:
                continue
            
            path = []
            curr = target
            while curr is not None:
                path.append(curr)
                curr = prev[curr]
            path.reverse()
            
            result[target] = (dist[target], path)
        
        return result

    def best_target(self) -> Optional[str]:
        if not self.node_state:
            return None
        
        best_node = max(self.node_state.items(), key=lambda x: x[1].height)
        return best_node[0]

    def next_hop(self, holding: str, target: Optional[str] = None) -> Optional[Tuple[str, str]]:
        if target is None:
            target = self.best_target()

        if target is None or holding == target:
            return None  # already at target — stay put, pay no fee
        
        paths = self.dijkstra(holding)
        
        if target not in paths or len(paths[target][1]) < 2:
            return self._random_hop(holding)
        
        path = paths[target][1]
        
        return (path[0], path[1])
    
    def _random_hop(self, holding: str) -> Optional[Tuple[str, str]]:
        available = []
        for (base, quote) in self.edge_state.keys():
            if base == holding:
                available.append((base, quote))
            elif quote == holding:
                available.append((quote, base))
        
        if not available:
            return None
        
        base, quote = random.choice(available)
        return (base, quote)

    def reinforce(self, base: str, quote: str, pnl: float):
        edge = (base, quote)
        if edge not in self.edge_state:
            return
        
        es = self.edge_state[edge]
        es.cumulative_pnl += pnl
        es.n_traversals += 1
        es.last_pnl = pnl
        
        if pnl > 0:
            es.conductance *= 1.02
            es.wins += 1
        else:
            es.conductance *= 0.90
            es.losses += 1
        
        es.conductance = np.clip(es.conductance, 0.01, 10.0)
        
        es.volatility_window.append(abs(es.velocity))
        if len(es.volatility_window) > 20:
            es.volatility_window.pop(0)

    def node_potentials(self) -> List[Tuple[float, str]]:
        return sorted([(ns.height, currency) for currency, ns in self.node_state.items()], reverse=True)

    def get_edge_stats(self) -> List[Dict]:
        stats = []
        for (base, quote), es in self.edge_state.items():
            accels = self._accel_history.get((base, quote), [])
            avg_accel = np.mean(accels) if accels else 0.0
            stats.append({
                'edge': f"{base}/{quote}",
                'conductance': es.conductance,
                'cumulative_pnl': es.cumulative_pnl,
                'n_traversals': es.n_traversals,
                'avg_accel': avg_accel,
                'last_pnl': es.last_pnl
            })
        return stats

    def get_node_stats(self) -> List[Dict]:
        stats = []
        for currency, ns in self.node_state.items():
            time_at_north_pct = (ns.time_at_north / ns.total_bars * 100) if ns.total_bars > 0 else 0.0
            stats.append({
                'currency': currency,
                'avg_height': ns.height,
                'time_at_north_pct': time_at_north_pct,
                'net_inflow': ns.net_inflow,
                'total_bars': ns.total_bars
            })
        return stats

    def hr_high_level_plan(self) -> Dict[str, str]:
        potentials = self.node_potentials()
        
        if not potentials:
            return {}
        
        median_height = np.median([p[0] for p in potentials])
        
        direction = {}
        for height, currency in potentials:
            if height > median_height * 1.5:
                direction[currency] = "north"
            elif height < median_height * 0.5:
                direction[currency] = "south"
            else:
                direction[currency] = "neutral"
        
        return direction

    def integrate_hrm_output(self, hrm_direction: Dict[str, str]):
        for edge, es in self.edge_state.items():
            base, quote = edge
            
            base_dir = hrm_direction.get(base, "neutral")
            quote_dir = hrm_direction.get(quote, "neutral")
            
            penalty = 0.0
            if base_dir == "south" and quote_dir == "north":
                penalty = 0.001
            elif base_dir == "north" and quote_dir == "south":
                penalty = -0.001
            
            self._hrm_penalty = getattr(self, '_hrm_penalty', {})
            self._hrm_penalty[edge] = penalty


if __name__ == "__main__":
    g = CoinGraph()
    n_bars = g.load()
    print(f"Loaded {len(g.nodes)} nodes, {len(g.edges)} edges, {n_bars} bars")
    
    print("\nInitial potentials (sorted):")
    for h, c in g.node_potentials()[:10]:
        print(f"  {c}: {h:.6f}")
