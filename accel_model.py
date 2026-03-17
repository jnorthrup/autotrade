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


class EdgeEncoder(nn.Module):
    """Encode per-edge fisheye features to 128-dim embedding."""
    def __init__(self, input_dim: int = 24, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, n_edges, 24)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x


class GraphAttentionLayer(nn.Module):
    """Single round of message passing with learned attention."""
    def __init__(self, edge_dim: int = 128, node_dim: int = 128):
        super().__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim

        # Attention score network: LeakyReLU(W_att @ [h_node || e_edge])
        self.W_att = nn.Linear(node_dim + edge_dim, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, node_emb, edge_emb, edge_list):
        """
        Args:
            node_emb: (n_nodes, node_dim) - node embeddings
            edge_emb: (n_edges, edge_dim) - edge embeddings
            edge_list: list of (base_idx, quote_idx) tuples

        Returns:
            new_node_emb: (n_nodes, node_dim) - aggregated node embeddings
        """
        n_nodes = node_emb.shape[0]

        # Build edge index mapping for fast lookup
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}

        # For each node, collect incident edge embeddings and compute attention
        node_updates = []
        for v in range(n_nodes):
            incident_edge_indices = []
            for e_idx, (base_idx, quote_idx) in enumerate(edge_list):
                if base_idx == v or quote_idx == v:
                    incident_edge_indices.append(e_idx)

            if not incident_edge_indices:
                node_updates.append(node_emb[v])
                continue

            # Gather edge embeddings incident to node v
            incident_emb = edge_emb[incident_edge_indices]  # (n_incident, edge_dim)

            # Compute attention scores
            h_v_repeat = node_emb[v:v+1].expand(len(incident_edge_indices), -1)  # (n_incident, node_dim)
            concat = torch.cat([h_v_repeat, incident_emb], dim=-1)  # (n_incident, node_dim + edge_dim)

            att_scores = self.W_att(concat)  # (n_incident, 1)
            att_scores = self.act(att_scores)
            att_weights = F.softmax(att_scores, dim=0)  # (n_incident, 1)

            # Aggregate: sum(a_i * e_i)
            aggregated = (att_weights * incident_emb).sum(dim=0)  # (edge_dim,)
            node_update = F.gelu(aggregated)
            node_updates.append(node_update)

        return torch.stack(node_updates, dim=0)


class GraphAttentionNetwork(nn.Module):
    """Full GAT for velocity prediction on trading graphs."""
    def __init__(self, edge_feature_dim: int = 24, edge_dim: int = 128,
                 node_dim: int = 128, n_message_passes: int = 2):
        super().__init__()

        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.n_message_passes = n_message_passes

        # Step 1: Edge Encoder
        self.edge_encoder = EdgeEncoder(input_dim=edge_feature_dim, hidden_dim=edge_dim)

        # Step 2: Message passing layers
        self.mp_layers = nn.ModuleList([
            GraphAttentionLayer(edge_dim=edge_dim, node_dim=node_dim)
            for _ in range(n_message_passes)
        ])

        # Step 3: Edge Decoder
        self.edge_decoder = nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, edge_features, edge_list, n_nodes):
        """
        Args:
            edge_features: (batch, n_edges, 24) - per-edge fisheye features
            edge_list: list of (base_idx, quote_idx) tuples
            n_nodes: number of nodes in graph

        Returns:
            velocities: (batch, n_edges) - predicted velocity per edge
        """
        batch_size = edge_features.shape[0]
        n_edges = edge_features.shape[1]

        # Step 1: Encode edges
        edge_emb = self.edge_encoder(edge_features)  # (batch, n_edges, 128)

        # Step 2: Initialize node embeddings (zeros or learnable)
        node_emb = torch.zeros(batch_size, n_nodes, self.node_dim, device=edge_features.device)

        # Step 3: Message passing (2 rounds)
        for mp_layer in self.mp_layers:
            # Process each batch item
            node_emb_new = []
            for b in range(batch_size):
                updated_nodes = mp_layer(node_emb[b], edge_emb[b], edge_list)
                node_emb_new.append(updated_nodes)
            node_emb = torch.stack(node_emb_new, dim=0)  # (batch, n_nodes, 128)

        # Step 4: Decode edges
        velocities = []
        for b in range(batch_size):
            edge_vels = []
            for ei, (base_idx, quote_idx) in enumerate(edge_list):
                h_base = node_emb[b, base_idx]  # (128,)
                h_quote = node_emb[b, quote_idx]  # (128,)
                e_emb = edge_emb[b, ei]  # (128,)

                concat = torch.cat([h_base, h_quote, e_emb], dim=-1)  # (384,)
                vel = self.edge_decoder(concat)  # (1,)
                edge_vels.append(vel)

            velocities.append(torch.cat(edge_vels, dim=0))  # (n_edges,)

        return torch.stack(velocities, dim=0)  # (batch, n_edges)


@dataclass
class HRMState:
    currency: str
    high_level_direction: str = "neutral"
    low_level_accuracy: float = 0.0
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)


class AccelModel:
    def __init__(self, n_edges: int, sequence_length: int = 8, learning_rate: float = 0.001,
                 edge_dim: int = 128, node_dim: int = 128, n_message_passes: int = 2,
                 y_depth: int = 200, x_pixels: int = 20, curvature: float = 2.0):
        self.n_edges = n_edges
        self.sequence_length = sequence_length
        self.y_depth = y_depth
        self.x_pixels = x_pixels
        self._lr = learning_rate
        self.edge_names: List[Tuple[str, str]] = []
        self.edge_to_idx: Dict[Tuple[str, str], int] = {}
        self.node_names: List[str] = []
        self.node_to_idx: Dict[str, int] = {}
        self.edge_list: List[Tuple[int, int]] = []  # (base_idx, quote_idx)

        self._default_curvature = curvature
        self._default_y_depth = y_depth
        self._default_x_pixels = x_pixels

        self._edge_curvature: Dict[Tuple[str, str], nn.Parameter] = {}
        self._edge_y_depth: Dict[Tuple[str, str], nn.Parameter] = {}
        self._edge_x_pixels: Dict[Tuple[str, str], nn.Parameter] = {}

        self._curve_optimizer = None

        # Graph Attention Network for velocity prediction
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.gat = GraphAttentionNetwork(
            edge_feature_dim=self.x_pixels + 4,  # 20 fisheye + 4 metadata
            edge_dim=edge_dim,
            node_dim=node_dim,
            n_message_passes=n_message_passes
        )

        self.velocity_optimizer = optim.Adam(self.gat.parameters(), lr=learning_rate)
        self.mse_criterion = nn.MSELoss()

        self.hrm_state: Dict[str, HRMState] = {}
        self._accel_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._close_buffer: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._direction_buffer: Dict[str, List[str]] = defaultdict(list)

        self._bar_count = 0
        self._high_level_update_freq = 10
        self._max_history = max(y_depth + 100, 500)

    def register_edges(self, edges: List[Tuple[str, str]]):
        """Build graph structure and initialize learnable fisheye parameters."""
        self.edge_names = edges

        # Build node set from edges
        node_set = set()
        for base, quote in edges:
            node_set.add(base)
            node_set.add(quote)

        self.node_names = sorted(list(node_set))
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}

        # Build edge list with node indices
        self.edge_list = []
        for base, quote in edges:
            base_idx = self.node_to_idx[base]
            quote_idx = self.node_to_idx[quote]
            self.edge_list.append((base_idx, quote_idx))
            self.edge_to_idx[(base, quote)] = len(self.edge_list) - 1

        # Initialize learnable fisheye parameters per edge
        for edge in edges:
            self._edge_curvature[edge] = nn.Parameter(torch.tensor(self._default_curvature, dtype=torch.float32))
            self._edge_y_depth[edge] = nn.Parameter(torch.tensor(float(self._default_y_depth), dtype=torch.float32))
            self._edge_x_pixels[edge] = nn.Parameter(torch.tensor(float(self._default_x_pixels), dtype=torch.float32))

        all_edge_params = list(self._edge_curvature.values()) + list(self._edge_y_depth.values()) + list(self._edge_x_pixels.values())
        self._curve_optimizer = optim.Adam(all_edge_params, lr=self._lr * 0.1)

    def _build_input_tensor(self, graph) -> torch.Tensor:
        """
        Build per-edge feature matrix: (1, n_edges, x_pixels + 4)
        Each edge gets: fisheye_values (x_pixels) + [curvature, y_depth/1000, x_pixels/100, fee_rate]
        """
        edge_features = []

        for edge in self.edge_names:
            curvature = float(torch.clamp(self._edge_curvature[edge], 0.1, 10.0).detach())
            y_depth = int(torch.clamp(self._edge_y_depth[edge], 10, 1000).detach())
            x_pixels = int(torch.clamp(self._edge_x_pixels[edge], 5, 50).detach())

            boundaries = fisheye_boundaries(y_depth, x_pixels, curvature)

            closes = self._close_buffer.get(edge, [])

            if len(closes) < 2:
                fisheye_values = [0.0] * self.x_pixels
            else:
                closes_array = np.array(closes[-y_depth:])
                fisheye_values = fisheye_sample(closes_array, boundaries)

            # Ensure we have exactly x_pixels values (pad or truncate)
            fisheye_values = fisheye_values[:self.x_pixels]
            while len(fisheye_values) < self.x_pixels:
                fisheye_values.append(0.0)

            feat = fisheye_values + [
                curvature,
                y_depth / 1000.0,
                x_pixels / 100.0,
                graph.fee_rate if hasattr(graph, 'fee_rate') else 0.001
            ]
            edge_features.append(feat)

        return torch.tensor(edge_features, dtype=torch.float32).unsqueeze(0)  # (1, n_edges, 24)

    def _encode_direction(self, direction: str) -> int:
        mapping = {"north": 0, "neutral": 1, "south": 2}
        return mapping.get(direction, 1)

    def high_level_plan(self, graph) -> Dict[str, str]:
        """Infer high-level direction per currency from graph structure."""
        self._bar_count += 1

        # Initialize HRM states if needed
        for node in self.node_names:
            if node not in self.hrm_state:
                self.hrm_state[node] = HRMState(currency=node)

        # Infer direction from node potentials (from graph.node_state)
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

    def predict(self, graph) -> Dict[Tuple[str, str], float]:
        """Predict per-edge velocity using GAT."""
        if not self.edge_names or not self.edge_list:
            return {}

        edge_features = self._build_input_tensor(graph)  # (1, n_edges, 24)

        with torch.no_grad():
            velocities = self.gat(edge_features, self.edge_list, len(self.node_names))  # (1, n_edges)
            preds = velocities.squeeze(0).tolist()

        if isinstance(preds, float):
            preds = [preds]

        predictions = {}
        for i, edge in enumerate(self.edge_names):
            if i < len(preds):
                predictions[edge] = preds[i]
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
        """Update GAT with actual velocities."""
        if not self.edge_names or not actual_accels or not self.edge_list:
            return None

        self.update_prices(graph, bar_idx)

        for edge, accel in actual_accels.items():
            if edge in self._accel_buffer:
                self._accel_buffer[edge].append(accel)
                if len(self._accel_buffer[edge]) > self._max_history:
                    self._accel_buffer[edge] = self._accel_buffer[edge][-self._max_history:]
            else:
                self._accel_buffer[edge].append(accel)

        edge_features = self._build_input_tensor(graph)  # (1, n_edges, 24)

        # Build target tensor
        target = []
        for edge in self.edge_names:
            target.append(actual_accels.get(edge, 0.0))

        target_t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)  # (1, n_edges)

        # Forward pass through GAT
        self.velocity_optimizer.zero_grad()
        velocities = self.gat(edge_features, self.edge_list, len(self.node_names))  # (1, n_edges)

        loss = self.mse_criterion(velocities, target_t)
        loss.backward()
        nn.utils.clip_grad_norm_(self.gat.parameters(), 1.0)
        self.velocity_optimizer.step()

        # Also update fisheye parameters
        if self._curve_optimizer is not None:
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

    def get_hrm_stats(self) -> List[Dict]:
        stats = []
        for currency, hs in self.hrm_state.items():
            stats.append({
                'currency': currency,
                'high_level_direction': hs.high_level_direction,
                'low_level_accuracy': hs.low_level_accuracy
            })
        return stats
