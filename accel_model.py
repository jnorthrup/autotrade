import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def fisheye_boundaries(y_depth: int, x_pixels: int, curvature: float) -> List[int]:
    """
    Non-linear bucket boundaries for fisheye compression.
    
    Args:
        y_depth: History candles (e.g., 200)
        x_pixels: Output pixels (e.g., 20)
        curvature: Power exponent
            - < 1: Fisheye focused on recent
            - = 1: Linear
            - > 1: More uniform
    
    Returns:
        List of bucket end indices (cumulative)
    """
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
    """
    Sample candles using fisheye bucket boundaries.
    Returns relative price (bucket_mean - current) / bucket_mean for each bucket.
    """
    if len(candles) == 0:
        return [0.0] * len(boundaries)
    
    results = []
    prev_idx = 0
    
    for i, boundary in enumerate(boundaries):
        bucket_start = prev_idx
        bucket_end = min(boundary, len(candles))
        
        if bucket_end <= bucket_start:
            results.append(0.0)
        else:
            bucket_candles = candles[bucket_start:bucket_end]
            bucket_mean = np.mean(bucket_candles)
            current_close = candles[-1] if len(candles) > 0 else bucket_mean
            
            if bucket_mean != 0:
                relative = (current_close - bucket_mean) / bucket_mean
            else:
                relative = 0.0
            results.append(relative)
        
        prev_idx = boundary
    
    return results


class AccelPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


@dataclass
class HRMState:
    currency: str
    high_level_direction: str = "neutral"
    low_level_accuracy: float = 0.0
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)


class AccelModel:
    def __init__(self, n_edges: int, sequence_length: int = 8, learning_rate: float = 0.001, hidden_dim: int = 64, 
                 y_depth: int = 200, x_pixels: int = 20, curvature: float = 2.0):
        self.n_edges = n_edges
        self.sequence_length = sequence_length
        self.y_depth = y_depth
        self.x_pixels = x_pixels
        self._lr = learning_rate
        self.edge_names: List[Tuple[str, str]] = []
        self.edge_to_idx: Dict[Tuple[str, str], int] = {}
        
        self._default_curvature = curvature
        self._default_y_depth = y_depth
        self._default_x_pixels = x_pixels
        
        self._edge_curvature: Dict[Tuple[str, str], nn.Parameter] = {}
        self._edge_y_depth: Dict[Tuple[str, str], nn.Parameter] = {}
        self._edge_x_pixels: Dict[Tuple[str, str], nn.Parameter] = {}
        
        self._curve_optimizer = None
        
        self.input_dim = 1  # placeholder - set after register_edges
        
        self._hidden_dim = hidden_dim
        self.high_model = AccelPredictor(self.input_dim, 3, self._hidden_dim)
        self.low_model = AccelPredictor(self.input_dim, n_edges, self._hidden_dim)
        
        self.high_optimizer = optim.Adam(self.high_model.parameters(), lr=learning_rate)
        self.low_optimizer = optim.Adam(self.low_model.parameters(), lr=learning_rate)
        
        self.high_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        
        self.hrm_state: Dict[str, HRMState] = {}
        self._accel_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._close_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._direction_buffer: Dict[str, List[str]] = defaultdict(list)
        
        self._bar_count = 0
        self._high_level_update_freq = 10
        self._max_history = max(y_depth + 100, 500)

    def register_edges(self, edges: List[Tuple[str, str]]):
        self.edge_names = edges
        
        self.input_dim = len(edges) * (self.x_pixels + 3)
        
        self.high_model = AccelPredictor(self.input_dim, 3, self._hidden_dim)
        self.low_model = AccelPredictor(self.input_dim, len(edges), self._hidden_dim)
        
        self.high_optimizer = optim.Adam(self.high_model.parameters(), lr=self._lr)
        self.low_optimizer = optim.Adam(self.low_model.parameters(), lr=self._lr)
        
        for edge in edges:
            self._edge_curvature[edge] = nn.Parameter(torch.tensor(self._default_curvature, dtype=torch.float32))
            self._edge_y_depth[edge] = nn.Parameter(torch.tensor(float(self._default_y_depth), dtype=torch.float32))
            self._edge_x_pixels[edge] = nn.Parameter(torch.tensor(float(self._default_x_pixels), dtype=torch.float32))
        
        all_edge_params = list(self._edge_curvature.values()) + list(self._edge_y_depth.values()) + list(self._edge_x_pixels.values())
        self._curve_optimizer = optim.Adam(all_edge_params, lr=self._lr * 0.1)

    def _build_input_tensor(self, graph) -> torch.Tensor:
        """
        Per-edge fisheye horizon: each trading pair learns its own curvature, y_depth, x_pixels.
        """
        all_inputs = []
        
        for edge in self.edge_names:
            curvature = float(torch.clamp(self._edge_curvature[edge], 0.1, 10.0).detach())
            y_depth = int(torch.clamp(self._edge_y_depth[edge], 10, 1000).detach())
            x_pixels = int(torch.clamp(self._edge_x_pixels[edge], 5, 50).detach())
            
            boundaries = fisheye_boundaries(y_depth, x_pixels, curvature)
            
            closes = self._close_buffer.get(edge, [])
            
            if len(closes) < 2:
                fisheye_values = [0.0] * x_pixels
            else:
                closes_array = np.array(closes[-y_depth:])
                fisheye_values = fisheye_sample(closes_array, boundaries)
            
            all_inputs.extend(fisheye_values)
            all_inputs.append(curvature)
            all_inputs.append(y_depth / 1000.0)
            all_inputs.append(x_pixels / 100.0)
            all_inputs.append(graph.fee_rate if hasattr(graph, 'fee_rate') else 0.001)
            
        return torch.tensor(all_inputs, dtype=torch.float32).unsqueeze(0)

    def _encode_direction(self, direction: str) -> int:
        mapping = {"north": 0, "neutral": 1, "south": 2}
        return mapping.get(direction, 1)

    def high_level_plan(self, graph) -> Dict[str, str]:
        self._bar_count += 1
        
        if self._bar_count % self._high_level_update_freq != 0:
            return {c: hs.high_level_direction for c, hs in self.hrm_state.items()}
        
        if not self.edge_names:
            return {}
        
        x = self._build_input_tensor(graph)
        
        with torch.no_grad():
            high_out = self.high_model(x)
            directions = torch.argmax(high_out, dim=1).squeeze().tolist()
            
        if isinstance(directions, int):
            directions = [directions]
        
        dir_map = {0: "north", 1: "neutral", 2: "south"}
        
        currencies = set()
        for base, quote in self.edge_names:
            currencies.add(base)
            currencies.add(quote)
        
        for i, c in enumerate(sorted(currencies)):
            if c in self.hrm_state:
                self.hrm_state[c].high_level_direction = dir_map.get(directions[0], "neutral")
                self._direction_buffer[c].append(self.hrm_state[c].high_level_direction)
                if len(self._direction_buffer[c]) > self.sequence_length:
                    self._direction_buffer[c].pop()
        
        return {c: hs.high_level_direction for c, hs in self.hrm_state.items()}

    def predict(self, graph) -> Dict[Tuple[str, str], float]:
        """Predict per-edge acceleration."""
        if not self.edge_names:
            return {}
        
        x = self._build_input_tensor(graph)
        
        with torch.no_grad():
            low_out = self.low_model(x)
            preds = low_out.squeeze().tolist()
        
        if isinstance(preds, float):
            preds = [preds]
        
        predictions = {}
        for i, edge in enumerate(self.edge_names):
            if i < len(preds):
                predictions[edge] = preds[i] / 100.0
            else:
                predictions[edge] = 0.0
        
        return predictions

    def update_prices(self, graph, bar_idx: int):
        """Update close price buffer from graph for fisheye sampling."""
        if bar_idx < 0:
            return
        
        for edge in self.edge_names:
            if edge in graph.edges:
                df = graph.edges[edge]
                if bar_idx < len(df):
                    close = float(df['close'].iloc[bar_idx])
                    if edge in self._close_buffer:
                        self._close_buffer[edge].append(close)
                        if len(self._close_buffer[edge]) > self._max_history:
                            self._close_buffer[edge] = self._close_buffer[edge][-self._max_history:]
                    else:
                        self._close_buffer[edge] = [close]

    def update(self, graph, actual_accels: Dict[Tuple[str, str], float], bar_idx: int = -1) -> Optional[float]:
        if not self.edge_names or not actual_accels:
            return None
        
        self.update_prices(graph, bar_idx)
        
        for edge, accel in actual_accels.items():
            if edge in self._accel_buffer:
                self._accel_buffer[edge].append(accel)
                if len(self._accel_buffer[edge]) > self._max_history:
                    self._accel_buffer[edge] = self._accel_buffer[edge][-self._max_history:]
            else:
                self._accel_buffer[edge].append(accel)
        
        x = self._build_input_tensor(graph)
        
        self.low_optimizer.zero_grad()
        low_out = self.low_model(x)
        
        target = []
        for edge in self.edge_names:
            target.append(actual_accels.get(edge, 0.0) * 100.0)
            
        target_t = torch.tensor(target, dtype=torch.float32)
        
        if low_out.shape[-1] != target_t.shape[0]:
            return None
        
        loss = self.mse_criterion(low_out.squeeze()[:len(target_t)], target_t)
        loss.backward()
        self.low_optimizer.step()
        
        self._curve_optimizer.zero_grad()
        curve_loss = 0.0
        for edge in self.edge_names:
            curve_loss = curve_loss + (
                self._edge_curvature[edge].pow(2) +
                (self._edge_y_depth[edge] - 200.0).pow(2) / 10000.0 +
                (self._edge_x_pixels[edge] - 20.0).pow(2) / 100.0
            )
        (curve_loss * 0.001).backward()
        self._curve_optimizer.step()
        
        for edge in self.edge_names:
            self._edge_curvature[edge].data.clamp_(0.1, 10.0)
            self._edge_y_depth[edge].data.clamp_(10, 1000)
            self._edge_x_pixels[edge].data.clamp_(5, 50)
        
        return loss.item()
                        
        return loss_val

    def get_hrm_stats(self) -> List[Dict]:
        stats = []
        for currency, hs in self.hrm_state.items():
            stats.append({
                'currency': currency,
                'high_level_direction': hs.high_level_direction,
                'low_level_accuracy': hs.low_level_accuracy
            })
        return stats
