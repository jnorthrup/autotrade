import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


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
    def __init__(self, n_edges: int, sequence_length: int = 8, learning_rate: float = 0.001, hidden_dim: int = 64, time_constant: float = 1.5):
        self.n_edges = n_edges
        self.sequence_length = sequence_length
        self.time_constant = time_constant
        self.edge_names: List[Tuple[str, str]] = []
        self.edge_to_idx: Dict[Tuple[str, str], int] = {}
        
        # input is sequence_length features + 1 volume feature per edge
        self.input_dim = n_edges * (sequence_length + 1)
        
        self._hidden_dim = hidden_dim
        self.high_model = AccelPredictor(self.input_dim, 3, self._hidden_dim)
        self.low_model = AccelPredictor(self.input_dim, n_edges, self._hidden_dim)
        
        self.high_optimizer = optim.Adam(self.high_model.parameters(), lr=learning_rate)
        self.low_optimizer = optim.Adam(self.low_model.parameters(), lr=learning_rate)
        
        self.high_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        
        self.hrm_state: Dict[str, HRMState] = {}
        self._accel_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._direction_buffer: Dict[str, List[str]] = defaultdict(list)
        
        self._bar_count = 0
        self._high_level_update_freq = 10
        # Determine how deep our max history buffer needs to be for the hyperbolic tail
        self._max_history = max(1000, int(1.5 ** self.sequence_length) + 10)

    def register_edges(self, edges: List[Tuple[str, str]]):
        self.edge_names = edges
        self.edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        currencies = set()
        for base, quote in edges:
            currencies.add(base)
            currencies.add(quote)
        
        for c in currencies:
            self.hrm_state[c] = HRMState(currency=c)

    def _build_input_tensor(self, graph) -> torch.Tensor:
        """
        Hyperbolic horizon array:
        Instead of linearly pulling the last N bars blindly, we generate
        `sequence_length` continuous buckets extending hyperbolically/exponentially into the past.
        This provides deep rearview context using true floating-point mathematics for a smooth compression curvature.
        """
        all_accels = []
        for edge in self.edge_names:
            accels = self._accel_buffer.get(edge, [])
            n_accels = len(accels)
            
            hyper_array = []
            for i in range(self.sequence_length):
                # Continuous vanishing point interpolation mapping into the past
                # Uses the tunable time_constant (curvature)
                lookback_float = self.time_constant ** i
                idx_lower = int(lookback_float)
                idx_upper = idx_lower + 1
                alpha = lookback_float - idx_lower
                
                if idx_lower >= n_accels:
                    if n_accels > 0:
                        hyper_array.append(accels[0])
                    else:
                        hyper_array.append(0.0)
                else:
                    # Linearly interpolate between the two discrete history points natively for FPU accuracy
                    val_lower = accels[-(idx_lower + 1)] if (idx_lower + 1) <= n_accels else accels[0]
                    val_upper = accels[-(idx_upper + 1)] if (idx_upper + 1) <= n_accels else val_lower
                    interpolated = (1.0 - alpha) * val_lower + alpha * val_upper
                    hyper_array.append(interpolated)
                    
            all_accels.extend(reversed(hyper_array))  # oldest to newest chronologically
            
            # Fetch the actual volume natively off the graph map to give +1 tensor 'Mass' context per edge
            vol = 0.0
            if edge in graph.edge_state:
                vol = graph.edge_state[edge].volume if hasattr(graph.edge_state[edge], 'volume') else 0.0
            elif (edge[1], edge[0]) in graph.edge_state:
                # Reverse mapping just in case mapping binds laterally
                vol = graph.edge_state[(edge[1], edge[0])].volume if hasattr(graph.edge_state[(edge[1], edge[0])], 'volume') else 0.0
            all_accels.append(vol)
            
        return torch.tensor(all_accels, dtype=torch.float32).unsqueeze(0)

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
                pred_val = preds[i] / 100.0  # Scale back down from the 100x target scaling
                predictions[edge] = pred_val
                predictions[(edge[1], edge[0])] = -pred_val
            else:
                predictions[edge] = 0.0
                predictions[(edge[1], edge[0])] = 0.0
        
        return predictions

    def update(self, graph, actual_accels: Dict[Tuple[str, str], float]) -> Optional[float]:
        if not self.edge_names or not actual_accels:
            return None
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
        for e in self.edge_names:
            target.append(actual_accels.get(e, 0.0) * 100.0) # Scale up naturally small percent velocity
            
        target_t = torch.tensor(target, dtype=torch.float32)
        
        if low_out.shape[-1] != target_t.shape[0]:
            return None
        
        loss = self.mse_criterion(low_out.squeeze()[:len(target_t)], target_t)
        loss.backward()
        self.low_optimizer.step()
        
        loss_val = loss.item()
        
        for edge, accel in actual_accels.items():
            if edge in self._accel_buffer and len(self._accel_buffer[edge]) >= 2:
                buf = self._accel_buffer[edge]
                if len(buf) >= 2:
                    pred = buf[-2] # This is just historical smoothing array, actual pred wasn't aligned properly
                    self.hrm_state[edge[0]].predictions.append(pred)
                    self.hrm_state[edge[0]].actuals.append(accel)
                    
                    if len(self.hrm_state[edge[0]].predictions) > 20:
                        self.hrm_state[edge[0]].predictions.pop(0)
                        self.hrm_state[edge[0]].actuals.pop(0)
                    
                    if len(self.hrm_state[edge[0]].predictions) > 0:
                        preds = self.hrm_state[edge[0]].predictions
                        acts = self.hrm_state[edge[0]].actuals
                        mse = np.mean([(p - a) ** 2 for p, a in zip(preds, acts)])
                        self.hrm_state[edge[0]].low_level_accuracy = 1.0 / (1.0 + mse)
                        
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
