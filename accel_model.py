import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
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
class PredictionFrame:
    edge: Tuple[str, str]
    predicted_fraction: float
    predicted_ptt: float
    predicted_stop: float
    fisheye: List[float]
    bar_idx: int


@dataclass
class HRMState:
    currency: str
    high_level_direction: str = "neutral"
    low_level_accuracy: float = 0.0


class EdgeModel(nn.Module):
    """Per-edge prediction from fisheye.
    
    Outputs per edge:
    - fraction: position size [0,1] (sigmoid)
    - PTT: profit target probability [0,1] (sigmoid)
    - STOP: stop loss probability [0,1] (sigmoid)
    """
    def __init__(self, x_pixels: int, h_dim: int = 16, z_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_pixels, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        self.fraction_head = nn.Linear(z_dim, 1)
        self.ptt_head = nn.Linear(z_dim, 1)
        self.stop_head = nn.Linear(z_dim, 1)
        
        nn.init.zeros_(self.fraction_head.weight)
        nn.init.zeros_(self.fraction_head.bias)
        nn.init.zeros_(self.ptt_head.weight)
        nn.init.zeros_(self.ptt_head.bias)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.zeros_(self.stop_head.bias)
    
    def forward(self, fisheye: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(fisheye)
        fraction = torch.sigmoid(self.fraction_head(z)).squeeze(-1)
        ptt = torch.sigmoid(self.ptt_head(z)).squeeze(-1)
        stop = torch.sigmoid(self.stop_head(z)).squeeze(-1)
        return fraction, ptt, stop


class AccelModel:
    def __init__(self, n_edges: int = 0, learning_rate: float = 0.001,
                 y_depth: int = 200, x_pixels: int = 20, curvature: float = 2.0,
                 h_dim: int = 16, z_dim: int = 8, prediction_depth: int = 1, **kwargs):
        self.y_depth = y_depth
        self.x_pixels = x_pixels
        self._lr = learning_rate
        self.curvature = curvature
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.prediction_depth = prediction_depth

        self.edge_names: List[Tuple[str, str]] = []
        self.node_names: List[str] = []
        self.node_to_idx: Dict[str, int] = {}

        self._model: Optional[EdgeModel] = None
        self._optimizer: Optional[optim.Optimizer] = None

        self.hrm_state: Dict[str, HRMState] = {}
        self._close_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._max_history = max(y_depth + 100, 500)
        
        self._prediction_queue: Dict[Tuple[str, str], deque] = {}

    def register_edges(self, edges: List[Tuple[str, str]]):
        self.edge_names = edges

        node_set = set()
        for base, quote in edges:
            node_set.add(base)
            node_set.add(quote)
        
        if "USD" in node_set:
            node_set.discard("USD")
            self.node_names = ["USD"] + sorted(node_set)
        else:
            self.node_names = sorted(node_set)
        
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}

        self._model = EdgeModel(self.x_pixels, self.h_dim, self.z_dim)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        
        for edge in edges:
            self._prediction_queue[edge] = deque(maxlen=self.prediction_depth)

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

    def predict(self, graph, bar_idx: int = -1) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        """Predict fraction, PTT, STOP for each edge.
        
        Returns: {edge: (fraction, ptt_prob, stop_prob)}
        """
        if not self.edge_names or self._model is None:
            return {}
        
        predictions = {}
        with torch.no_grad():
            for edge in self.edge_names:
                fisheye = torch.tensor(self._get_fisheye(edge), dtype=torch.float32).unsqueeze(0)
                fraction, ptt, stop = self._model(fisheye)
                f, p, s = float(fraction), float(ptt), float(stop)
                predictions[edge] = (f, p, s)
                
                self._prediction_queue[edge].append(PredictionFrame(
                    edge=edge,
                    predicted_fraction=f,
                    predicted_ptt=p,
                    predicted_stop=s,
                    fisheye=self._get_fisheye(edge),
                    bar_idx=bar_idx
                ))
        
        return predictions

    def update(self, graph, actual_accels: Dict[Tuple[str, str], float], bar_idx: int = -1,
               actual_velocities: Optional[Dict[Tuple[str, str], float]] = None,
               hit_ptt: Optional[Dict[Tuple[str, str], bool]] = None,
               hit_stop: Optional[Dict[Tuple[str, str], bool]] = None) -> Optional[float]:
        """Train on fraction, PTT, STOP predictions.
        
        hit_ptt/stop: {edge: bool} - did edge cross band?
        """
        if not self.edge_names or self._model is None or self._optimizer is None:
            return None

        self.update_prices(graph, bar_idx)
        
        if hit_ptt is None or hit_stop is None:
            return None

        self._optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0)
        n_scored = 0
        
        for edge in self.edge_names:
            queue = self._prediction_queue.get(edge)
            if not queue or len(queue) < self.prediction_depth:
                continue
            
            frame = queue[0]
            
            # PTT/STOP targets
            ptt_target = 1.0 if hit_ptt.get(edge, False) else 0.0
            stop_target = 1.0 if hit_stop.get(edge, False) else 0.0
            
            # Fraction target: 1.0 = trade (PTT hit), 0.0 = avoid (STOP hit), 0.5 = neutral
            if hit_ptt.get(edge, False):
                frac_target = 1.0
            elif hit_stop.get(edge, False):
                frac_target = 0.0
            else:
                frac_target = 0.5
            
            fisheye = torch.tensor(frame.fisheye, dtype=torch.float32).unsqueeze(0)
            pred_frac, pred_ptt, pred_stop = self._model(fisheye)
            
            frac_loss = nn.functional.binary_cross_entropy(pred_frac, torch.tensor([frac_target]))
            ptt_loss = nn.functional.binary_cross_entropy(pred_ptt, torch.tensor([ptt_target]))
            stop_loss = nn.functional.binary_cross_entropy(pred_stop, torch.tensor([stop_target]))
            
            total_loss = total_loss + frac_loss + ptt_loss + stop_loss
            n_scored += 1

        if n_scored > 0:
            loss = total_loss / n_scored
            loss.backward()
            self._optimizer.step()
            return loss.item()
        return 0.0

    def high_level_plan(self, graph) -> Dict[str, str]:
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

        return dir_map

    def get_hrm_stats(self) -> List[Dict]:
        return [
            {'currency': currency, 'high_level_direction': hs.high_level_direction,
             'low_level_accuracy': hs.low_level_accuracy}
            for currency, hs in self.hrm_state.items()
        ]

    def save(self, path: str = "model_weights.pt"):
        if self._model is None:
            return
        state = {
            'model': self._model.state_dict(),
            'x_pixels': self.x_pixels,
            'y_depth': self.y_depth,
            'curvature': self.curvature,
            'h_dim': self.h_dim,
            'z_dim': self.z_dim,
            'prediction_depth': self.prediction_depth,
        }
        torch.save(state, path)

    def load(self, path: str = "model_weights.pt"):
        if not Path(path).exists():
            return
        state = torch.load(path, weights_only=True)
        if self._model is not None and 'model' in state:
            self._model.load_state_dict(state['model'])
