import numpy as np
from collections import deque
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
class PredictionFrame:
    edge: Tuple[str, str]
    predicted_velocity: float
    fisheye: List[float]
    node_heights: np.ndarray
    bar_idx: int


@dataclass
class HRMState:
    currency: str
    high_level_direction: str = "neutral"
    low_level_accuracy: float = 0.0


class Encoder(nn.Module):
    def __init__(self, n_nodes: int, h_dim: int = 16, z_dim: int = 8):
        super().__init__()
        # Input is 2*n_nodes: [heights, usd_prices]
        self.fc1 = nn.Linear(n_nodes * 2, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
    
    def forward(self, node_heights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(node_heights))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, x_pixels: int, z_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(x_pixels + z_dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, fisheye: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fisheye, z], dim=-1)
        return self.fc(x).squeeze(-1)


class HRM_VAE(nn.Module):
    def __init__(self, n_nodes: int, x_pixels: int, h_dim: int = 16, z_dim: int = 8):
        super().__init__()
        self.encoder = Encoder(n_nodes, h_dim, z_dim)
        self.decoder = Decoder(x_pixels, z_dim)
        self.z_dim = z_dim
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, node_heights: torch.Tensor, fisheye: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(node_heights)
        z = self.reparameterize(mu, logvar)
        velocity = self.decoder(fisheye, z)
        return velocity, mu, logvar


class AccelModel:
    def __init__(self, n_edges: int = 0, sequence_length: int = 8, learning_rate: float = 0.001,
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

        self._model: Optional[HRM_VAE] = None
        self._optimizer: Optional[optim.Optimizer] = None

        self.hrm_state: Dict[str, HRMState] = {}
        self._close_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._max_history = max(y_depth + 100, 500)
        
        self._kl_weight = 0.01
        
        self._prediction_queue: Dict[Tuple[str, str], deque] = {}

    def register_edges(self, edges: List[Tuple[str, str]]):
        self.edge_names = edges

        node_set = set()
        for base, quote in edges:
            node_set.add(base)
            node_set.add(quote)
        
        # USD is countercoin-0 (ground truth for PnL)
        if "USD" in node_set:
            node_set.discard("USD")
            self.node_names = ["USD"] + sorted(node_set)
        else:
            self.node_names = sorted(node_set)
        
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}
        n_nodes = len(self.node_names)

        self._model = HRM_VAE(n_nodes, self.x_pixels, self.h_dim, self.z_dim)
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

    def _get_node_heights(self, graph) -> np.ndarray:
        heights = np.zeros(len(self.node_names), dtype=np.float32)
        usd_prices = np.zeros(len(self.node_names), dtype=np.float32)
        usd_prices[0] = 1.0  # USD = 1 USD
        
        for i, node in enumerate(self.node_names):
            if node == "USD":
                heights[i] = 0.0
                continue
            if hasattr(graph, 'node_state') and node in graph.node_state:
                heights[i] = float(graph.node_state[node].height)
            
            # USD price: direct pair
            if (node, "USD") in graph.edges:
                df = graph.edges[(node, "USD")]
                if len(df) > 0:
                    usd_prices[i] = float(df['close'].iloc[-1])
            elif ("USD", node) in graph.edges:
                df = graph.edges[("USD", node)]
                if len(df) > 0:
                    usd_prices[i] = 1.0 / float(df['close'].iloc[-1])
        
        # Fill missing USD prices via 2-hop paths: BTC -> X -> USD
        for i, node in enumerate(self.node_names):
            if usd_prices[i] == 0.0:
                for j, intermediary in enumerate(self.node_names):
                    if usd_prices[j] == 0.0 or intermediary == node:
                        continue
                    # Check node -> intermediary -> USD
                    if (node, intermediary) in graph.edges and usd_prices[j] > 0:
                        df = graph.edges[(node, intermediary)]
                        if len(df) > 0:
                            price_to_intermediary = float(df['close'].iloc[-1])
                            usd_prices[i] = price_to_intermediary * usd_prices[j]
                            break
                    elif (intermediary, node) in graph.edges and usd_prices[j] > 0:
                        df = graph.edges[(intermediary, node)]
                        if len(df) > 0:
                            price_from_intermediary = float(df['close'].iloc[-1])
                            usd_prices[i] = usd_prices[j] / price_from_intermediary
                            break
        
        return np.concatenate([heights, usd_prices])

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

    def predict(self, graph, bar_idx: int = -1) -> Dict[Tuple[str, str], float]:
        if not self.edge_names or self._model is None:
            return {}
        
        node_heights_arr = self._get_node_heights(graph)
        node_heights = torch.from_numpy(node_heights_arr).unsqueeze(0)
        
        predictions = {}
        with torch.no_grad():
            for edge in self.edge_names:
                fisheye = torch.tensor(self._get_fisheye(edge), dtype=torch.float32).unsqueeze(0)
                vel, _, _ = self._model(node_heights, fisheye)
                predicted_velocity = float(vel)
                predictions[edge] = predicted_velocity
                
                self._prediction_queue[edge].append(PredictionFrame(
                    edge=edge,
                    predicted_velocity=predicted_velocity,
                    fisheye=self._get_fisheye(edge),
                    node_heights=node_heights_arr.copy(),
                    bar_idx=bar_idx
                ))
        
        return predictions

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
        if not self.edge_names or self._model is None or self._optimizer is None:
            return None

        self.update_prices(graph, bar_idx)
        targets = actual_velocities if actual_velocities else actual_accels

        self._optimizer.zero_grad()
        
        total_pred_loss = torch.tensor(0.0)
        total_kl_loss = torch.tensor(0.0)
        n_scored = 0
        
        for edge in self.edge_names:
            queue = self._prediction_queue.get(edge)
            if not queue or len(queue) < self.prediction_depth:
                continue
            
            frame = queue[0]
            actual = targets.get(edge, 0.0)
            
            node_heights = torch.from_numpy(frame.node_heights).unsqueeze(0)
            fisheye = torch.tensor(frame.fisheye, dtype=torch.float32).unsqueeze(0)
            
            pred, mu, logvar = self._model(node_heights, fisheye)
            
            pred_loss = self._hyperbolic_loss(pred, actual)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_pred_loss = total_pred_loss + pred_loss
            total_kl_loss = total_kl_loss + kl_loss
            n_scored += 1
            
            base, _ = edge
            if base in self.hrm_state:
                hs = self.hrm_state[base]
                hs.low_level_accuracy = hs.low_level_accuracy * 0.99 + (
                    1.0 if (float(pred.detach()) > 0) == (actual > 0) else 0.0
                ) * 0.01

        if n_scored > 0:
            loss = total_pred_loss / n_scored + self._kl_weight * total_kl_loss / n_scored
            loss.backward()
            self._optimizer.step()
            return float(loss)
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
