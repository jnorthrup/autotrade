import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
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


class AccelModel:
    def __init__(self, n_edges: int, sequence_length: int = 8, learning_rate: float = 0.001,
                 y_depth: int = 200, x_pixels: int = 20, curvature: float = 2.0, **kwargs):
        self.n_edges = n_edges
        self.sequence_length = sequence_length
        self.y_depth = y_depth
        self.x_pixels = x_pixels
        self._lr = learning_rate
        self.edge_names: List[Tuple[str, str]] = []
        self.edge_to_idx: Dict[Tuple[str, str], int] = {}
        self.node_names: List[str] = []
        self.node_to_idx: Dict[str, int] = {}
        self.edge_list: List[Tuple[int, int]] = []

        self._default_curvature = curvature
        self._default_y_depth = y_depth
        self._default_x_pixels = x_pixels

        self._edge_curvature: Dict[Tuple[str, str], nn.Parameter] = {}
        self._edge_y_depth: Dict[Tuple[str, str], nn.Parameter] = {}
        self._edge_x_pixels: Dict[Tuple[str, str], nn.Parameter] = {}
        self._curve_optimizer = None

        self.hrm_state: Dict[str, HRMState] = {}
        self._accel_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._close_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._direction_buffer: Dict[str, List[str]] = defaultdict(list)

        self._bar_count = 0
        self._high_level_update_freq = 10
        self._max_history = max(y_depth + 100, 500)

    def register_edges(self, edges: List[Tuple[str, str]]):
        self.edge_names = edges

        node_set = set()
        for base, quote in edges:
            node_set.add(base)
            node_set.add(quote)

        self.node_names = sorted(list(node_set))
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}

        self.edge_list = []
        for base, quote in edges:
            base_idx = self.node_to_idx[base]
            quote_idx = self.node_to_idx[quote]
            self.edge_list.append((base_idx, quote_idx))
            self.edge_to_idx[(base, quote)] = len(self.edge_list) - 1

        for edge in edges:
            self._edge_curvature[edge] = nn.Parameter(torch.tensor(self._default_curvature, dtype=torch.float32))
            self._edge_y_depth[edge] = nn.Parameter(torch.tensor(float(self._default_y_depth), dtype=torch.float32))
            self._edge_x_pixels[edge] = nn.Parameter(torch.tensor(float(self._default_x_pixels), dtype=torch.float32))

        all_edge_params = list(self._edge_curvature.values()) + list(self._edge_y_depth.values()) + list(self._edge_x_pixels.values())
        self._curve_optimizer = optim.Adam(all_edge_params, lr=self._lr * 0.1)

    def _get_fisheye(self, edge: Tuple[str, str]) -> List[float]:
        curvature = float(torch.clamp(self._edge_curvature[edge], 0.1, 10.0).detach())
        y_depth = int(torch.clamp(self._edge_y_depth[edge], 10, 1000).detach())
        x_pixels = int(torch.clamp(self._edge_x_pixels[edge], 5, 50).detach())
        boundaries = fisheye_boundaries(y_depth, x_pixels, curvature)
        closes = self._close_buffer.get(edge, [])
        if len(closes) < 2:
            return [0.0] * self.x_pixels
        closes_array = np.array(closes[-y_depth:])
        values = fisheye_sample(closes_array, boundaries)
        values = values[:self.x_pixels]
        while len(values) < self.x_pixels:
            values.append(0.0)
        return values

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
        if not self.edge_names:
            return {}
        predictions = {}
        for edge in self.edge_names:
            fisheye = self._get_fisheye(edge)
            # Most-recent bucket is fisheye[-1]; momentum signal
            predictions[edge] = fisheye[-1] if fisheye else 0.0
        return predictions

    def update(self, graph, actual_accels: Dict[Tuple[str, str], float], bar_idx: int = -1) -> Optional[float]:
        if not self.edge_names or not actual_accels:
            return None

        self.update_prices(graph, bar_idx)

        for i, edge in enumerate(self.edge_names):
            base, quote = edge
            actual = actual_accels.get(edge, 0.0)
            fisheye = self._get_fisheye(edge)
            pred = fisheye[-1] if fisheye else 0.0

            self._accel_buffer[edge].append(actual)
            if len(self._accel_buffer[edge]) > self._max_history:
                self._accel_buffer[edge] = self._accel_buffer[edge][-self._max_history:]

            if base in self.hrm_state:
                self.hrm_state[base].predictions.append(pred)
                self.hrm_state[base].actuals.append(actual)
                if len(self.hrm_state[base].predictions) > 100:
                    self.hrm_state[base].predictions.pop(0)
                    self.hrm_state[base].actuals.pop(0)
                if len(self.hrm_state[base].predictions) > 10:
                    with np.errstate(invalid='ignore'):
                        corr = np.corrcoef(self.hrm_state[base].predictions, self.hrm_state[base].actuals)[0, 1]
                    self.hrm_state[base].low_level_accuracy = 0.0 if np.isnan(corr) else corr

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
