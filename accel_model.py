import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def fisheye_boundaries(y_depth: int, x_pixels: int, curvature: float) -> List[int]:
    if x_pixels <= 1:
        return [y_depth]
    boundaries = []
    prev_boundary = 0
    for i in range(x_pixels):
        t = i / (x_pixels - 1)
        warped = t ** curvature
        boundary = int(y_depth * warped)
        boundary = max(boundary, prev_boundary + 1)
        boundaries.append(boundary)
        prev_boundary = boundary
    return boundaries


def fisheye_sample(candles: np.ndarray, boundaries: List[int]) -> List[float]:
    if len(candles) == 0:
        return [0.0] * len(boundaries)
    results = []
    prev_idx = 0
    for boundary in boundaries:
        bucket_start = prev_idx
        bucket_end = min(boundary, len(candles))
        if bucket_end <= bucket_start:
            results.append(0.0)
        else:
            bucket_candles = candles[bucket_start:bucket_end]
            bucket_mean = np.mean(bucket_candles)
            current_close = candles[-1]
            results.append((current_close - bucket_mean) / bucket_mean if bucket_mean != 0 else 0.0)
        prev_idx = boundary
    return results


@dataclass
class HRMState:
    currency: str
    high_level_direction: str = "neutral"
    low_level_accuracy: float = 0.0
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)


class HighLevelModule(nn.Module):
    """Shared module that sees the whole graph — encodes competitive landscape."""
    def __init__(self, n_nodes: int, h_dim: int = 8):
        super().__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim
        self.embed = nn.Linear(n_nodes, n_nodes * h_dim)
        nn.init.xavier_uniform_(self.embed.weight, gain=0.1)
        nn.init.zeros_(self.embed.bias)

    def forward(self, node_heights: torch.Tensor) -> torch.Tensor:
        x = self.embed(node_heights)
        return x.view(-1, self.n_nodes, self.h_dim)


class LowLevelModule(nn.Module):
    """Per-edge head: fisheye + both coin embeddings -> velocity."""
    def __init__(self, x_pixels: int, h_dim: int = 8):
        super().__init__()
        self.head = nn.Linear(x_pixels + 2 * h_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, fisheye: torch.Tensor, base_embed: torch.Tensor, quote_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fisheye, base_embed, quote_embed], dim=-1)
        return self.head(x).squeeze(-1)


class AccelModel:
    def __init__(self, n_edges: int, sequence_length: int = 8, learning_rate: float = 0.001,
                 y_depth: int = 200, x_pixels: int = 20, curvature: float = 2.0, h_dim: int = 8, **kwargs):
        self.n_edges = n_edges
        self.sequence_length = sequence_length
        self.y_depth = y_depth
        self.x_pixels = x_pixels
        self._lr = learning_rate
        self.curvature = curvature
        self.h_dim = h_dim

        self.edge_names: List[Tuple[str, str]] = []
        self.edge_to_idx: Dict[Tuple[str, str], int] = {}
        self.node_names: List[str] = []
        self.node_to_idx: Dict[str, int] = {}
        self.edge_list: List[Tuple[int, int]] = []

        self._high_level: Optional[HighLevelModule] = None
        self._low_level: Dict[Tuple[str, str], LowLevelModule] = {}
        self._optimizer: Optional[optim.Optimizer] = None

        self.hrm_state: Dict[str, HRMState] = {}
        self._accel_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._close_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._direction_buffer: Dict[str, List[str]] = defaultdict(list)

        self._bar_count = 0
        self._high_level_update_freq = 10
        self._max_history = max(y_depth + 100, 500)

        self._node_heights_buffer: Dict[str, List[float]] = defaultdict(list)

    def register_edges(self, edges: List[Tuple[str, str]]):
        self.edge_names = edges

        node_set = set()
        for base, quote in edges:
            node_set.add(base)
            node_set.add(quote)
        self.node_names = sorted(list(node_set))
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}
        n_nodes = len(self.node_names)

        self.edge_list = []
        for base, quote in edges:
            self.edge_list.append((self.node_to_idx[base], self.node_to_idx[quote]))
            self.edge_to_idx[(base, quote)] = len(self.edge_list) - 1

        self._high_level = HighLevelModule(n_nodes, self.h_dim)

        for edge in edges:
            if edge not in self._low_level:
                self._low_level[edge] = LowLevelModule(self.x_pixels, self.h_dim)

        all_params = list(self._high_level.parameters())
        for ll in self._low_level.values():
            all_params.extend(ll.parameters())
        self._optimizer = optim.Adam(all_params, lr=self._lr)

    def _get_fisheye(self, edge: Tuple[str, str]) -> List[float]:
        closes = self._close_buffer.get(edge, [])
        if len(closes) < 2:
            return [0.0] * self.x_pixels
        boundaries = fisheye_boundaries(self.y_depth, self.x_pixels, self.curvature)
        closes_array = np.array(closes[-self.y_depth:])
        values = fisheye_sample(closes_array, boundaries)
        values = values[:self.x_pixels]
        while len(values) < self.x_pixels:
            values.append(0.0)
        return values

    def _get_node_heights(self, graph) -> torch.Tensor:
        heights = np.zeros(len(self.node_names), dtype=np.float32)
        for i, node in enumerate(self.node_names):
            if hasattr(graph, 'node_state') and node in graph.node_state:
                heights[i] = float(graph.node_state[node].height)
            else:
                heights[i] = 0.0
        return torch.from_numpy(heights)

    def _forward(self, edge: Tuple[str, str], node_embeddings: torch.Tensor) -> torch.Tensor:
        base, quote = edge
        base_idx = self.node_to_idx.get(base, 0)
        quote_idx = self.node_to_idx.get(quote, 0)
        base_embed = node_embeddings[0, base_idx, :]
        quote_embed = node_embeddings[0, quote_idx, :]
        fisheye = torch.tensor(self._get_fisheye(edge), dtype=torch.float32)
        return self._low_level[edge](fisheye.unsqueeze(0), base_embed.unsqueeze(0), quote_embed.unsqueeze(0))

    def update_prices(self, graph, bar_idx: int):
        if bar_idx < 0:
            return
        for edge in self.edge_names:
            if edge in graph.edges:
                df = graph.edges[edge]
                if bar_idx < len(df):
                    close = float(df['close'].iloc[bar_idx])
                    buf = self._close_buffer[edge]
                    buf.append(close)
                    if len(buf) > self._max_history:
                        self._close_buffer[edge] = buf[-self._max_history:]

    def predict(self, graph) -> Dict[Tuple[str, str], float]:
        if not self.edge_names or self._high_level is None:
            return {}
        with torch.no_grad():
            node_heights = self._get_node_heights(graph)
            node_embeddings = self._high_level(node_heights.unsqueeze(0))
            return {edge: float(self._forward(edge, node_embeddings)) for edge in self.edge_names}

    def _hyperbolic_loss(self, pred: torch.Tensor, actual: float) -> torch.Tensor:
        pred_val = float(pred.detach())
        pred_sign = 1.0 if pred_val > 0 else -1.0 if pred_val < 0 else 0.0
        actual_sign = 1.0 if actual > 0 else -1.0 if actual < 0 else 0.0
        
        raw_mse = (pred - actual) ** 2
        
        if pred_sign != 0 and actual_sign != 0:
            if pred_sign == actual_sign:
                return raw_mse * 0.5
            else:
                return raw_mse * 4.0
        return raw_mse

    def update(self, graph, actual_accels: Dict[Tuple[str, str], float], bar_idx: int = -1,
               actual_velocities: Optional[Dict[Tuple[str, str], float]] = None) -> Optional[float]:
        if not self.edge_names or not actual_accels or self._optimizer is None or self._high_level is None:
            return None

        self.update_prices(graph, bar_idx)

        targets = actual_velocities if actual_velocities else actual_accels

        self._optimizer.zero_grad()
        
        node_heights = self._get_node_heights(graph)
        node_embeddings = self._high_level(node_heights.unsqueeze(0))
        
        total_loss = torch.tensor(0.0)
        n = 0
        for edge in self.edge_names:
            actual = targets.get(edge, 0.0)
            pred = self._forward(edge, node_embeddings)
            loss = self._hyperbolic_loss(pred, actual)
            total_loss = total_loss + loss
            n += 1

            base, _ = edge
            if base in self.hrm_state:
                hs = self.hrm_state[base]
                hs.predictions.append(float(pred.detach()))
                hs.actuals.append(actual)
                if len(hs.predictions) > 100:
                    hs.predictions.pop(0)
                    hs.actuals.pop(0)
                if len(hs.predictions) > 10:
                    with np.errstate(invalid='ignore'):
                        corr = np.corrcoef(hs.predictions, hs.actuals)[0, 1]
                    hs.low_level_accuracy = 0.0 if np.isnan(corr) else corr

        if n > 0:
            loss = total_loss / n
            loss.backward()
            self._optimizer.step()
            return float(loss)
        return 0.0

    def high_level_plan(self, graph) -> Dict[str, str]:
        self._bar_count += 1
        for node in self.node_names:
            if node not in self.hrm_state:
                self.hrm_state[node] = HRMState(currency=node)

        dir_map = {}
        for node in self.node_names:
            if hasattr(graph, 'node_state') and node in graph.node_state:
                height = graph.node_state[node].height
                if height > 0.0001:
                    dir_map[node] = "north"
                elif height < -0.0001:
                    dir_map[node] = "south"
                else:
                    dir_map[node] = "neutral"
            else:
                dir_map[node] = "neutral"

            self.hrm_state[node].high_level_direction = dir_map[node]
            self._direction_buffer[node].append(dir_map[node])
            if len(self._direction_buffer[node]) > self.sequence_length:
                self._direction_buffer[node].pop(0)

        return {c: hs.high_level_direction for c, hs in self.hrm_state.items()}

    def get_hrm_stats(self) -> List[Dict]:
        return [
            {'currency': currency, 'high_level_direction': hs.high_level_direction,
             'low_level_accuracy': hs.low_level_accuracy}
            for currency, hs in self.hrm_state.items()
        ]

    def save(self, path: str = "model_weights.pt"):
        if self._high_level is None:
            return
        state = {
            'high_level': self._high_level.state_dict(),
            'low_level': {str(k): v.state_dict() for k, v in self._low_level.items()},
            'x_pixels': self.x_pixels,
            'y_depth': self.y_depth,
            'curvature': self.curvature,
            'h_dim': self.h_dim,
        }
        torch.save(state, path)
        print(f"Saved model weights to {path}")

    def load(self, path: str = "model_weights.pt"):
        if not Path(path).exists():
            return
        state = torch.load(path, weights_only=True)
        
        if 'high_level' in state and self._high_level is not None:
            self._high_level.load_state_dict(state['high_level'])
        
        for k_str, sd in state.get('low_level', {}).items():
            inner = k_str.strip("()").replace("'", "")
            parts = [p.strip() for p in inner.split(",")]
            edge = (parts[0], parts[1])
            if edge in self._low_level:
                self._low_level[edge].load_state_dict(sd)
        print(f"Loaded model weights from {path}")
