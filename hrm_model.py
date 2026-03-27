"""
HRMEdgePredictor: Hierarchical Reasoning Model adapted for edge prediction.

Adapts HRM's two-level reasoning (H/L cycles) to predict fraction, PTT, STOP
per edge using fisheye-compressed temporal features.

Rotation Growth:
- Quadrant layout (0° preserved, 90°/180°/270° added):
    ┌────────┬────────┐
    │  0°    │ 180°   │
    ├────────┼────────┤
    │  90°   │ 270°   │
    └────────┴────────┘
- Growth factor: always 4×
- Square cube progression: hidden_size grows in sync with layers
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Keep graph_showdown self-contained instead of reaching into the nested HRM repo.
from layers_fallback import rms_norm, SwiGLU, Attention, RotaryEmbedding


def trunc_normal_init_(tensor, std=1.0):
    return nn.init.normal_(tensor, mean=0.0, std=std)


# ── Rotation utilities ────────────────────────────────────────────────────────

def rotate_90(W: Tensor) -> Tensor:
    """90° counter-clockwise rotation in weight space."""
    return torch.rot90(W, k=1, dims=(-2, -1))

def rotate_180(W: Tensor) -> Tensor:
    """180° rotation."""
    return torch.rot90(W, k=2, dims=(-2, -1))

def rotate_270(W: Tensor) -> Tensor:
    """270° rotation (90° clockwise)."""
    return torch.rot90(W, k=3, dims=(-2, -1))


# ── Square cube sizes ──────────────────────────────────────────────────────────

SQUARE_CUBE_SIZES = [4, 16, 64, 256]  # hidden_size progression


# ── HRM building blocks ───────────────────────────────────────────────────────

class FisheyeEmbedding(nn.Module):
    """Project fisheye features to hidden_size."""
    def __init__(self, x_pixels: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(x_pixels, hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, fisheye: Tensor) -> Tensor:
        # fisheye: (batch, x_pixels) → (batch, 1, hidden_size)
        x = self.proj(fisheye)
        return x.unsqueeze(1)  # seq_len=1 for per-edge


class HRMBlock(nn.Module):
    """Single transformer block: self-attn + SwiGLU MLP, post-norm."""
    def __init__(self, hidden_size: int, num_heads: int, expansion: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.norm_eps = 1e-5

    def forward(self, cos_sin, hidden_states: Tensor) -> Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class HRMReasoningModule(nn.Module):
    """Stack of HRM blocks with input injection."""
    def __init__(self, hidden_size: int, num_heads: int, n_layers: int, expansion: float = 4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            HRMBlock(hidden_size, num_heads, expansion)
            for _ in range(n_layers)
        ])

    def forward(self, hidden_states: Tensor, input_injection: Tensor, cos_sin) -> Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states


@dataclass
class HRMEdgeCarry:
    """Hierarchical carry state between H/L cycles."""
    z_H: Tensor
    z_L: Tensor


class HRMEdgePredictor(nn.Module):
    """
    Hierarchical Reasoning Model adapted for edge prediction.

    Replaces HRM's token embedding + LM head with:
    - FisheyeEmbedding: x_pixels → hidden_size
    - 3 prediction heads: fraction, PTT, STOP

    Architecture mirrors HRM:
    - H_level: slow, abstract planning (high-level reasoning)
    - L_level: rapid, detailed computation (low-level reasoning)
    - H_cycles × L_cycles per forward pass
    """
    def __init__(
        self,
        x_pixels: int = 20,
        hidden_size: int = 64,
        num_heads: int = 4,
        H_layers: int = 2,
        L_layers: int = 2,
        H_cycles: int = 2,
        L_cycles: int = 2,
        expansion: float = 4.0,
        rope_theta: float = 10000.0,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.x_pixels = x_pixels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.expansion = expansion

        # Input embedding from fisheye features
        self.embed = FisheyeEmbedding(x_pixels, hidden_size)

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
        )

        # H-level (high) and L-level (low) reasoning modules
        self.H_level = HRMReasoningModule(hidden_size, num_heads, H_layers, expansion)
        self.L_level = HRMReasoningModule(hidden_size, num_heads, L_layers, expansion)

        # Initial carry states
        self.register_buffer(
            "H_init",
            trunc_normal_init_(torch.empty(hidden_size, dtype=torch.bfloat16), std=1),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(hidden_size, dtype=torch.bfloat16), std=1),
            persistent=True,
        )

        # Prediction heads (operate on final z_H representation)
        self.fraction_head = nn.Linear(hidden_size, 1)
        self.ptt_head = nn.Linear(hidden_size, 1)
        self.stop_head = nn.Linear(hidden_size, 1)

        # Zero-init prediction heads for fast bootstrapping
        for head in [self.fraction_head, self.ptt_head, self.stop_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def init_carry(self, batch_size: int, device: torch.device) -> HRMEdgeCarry:
        """Initialize empty carry states."""
        return HRMEdgeCarry(
            z_H=torch.empty(batch_size, 1, self.hidden_size, device=device, dtype=torch.bfloat16),
            z_L=torch.empty(batch_size, 1, self.hidden_size, device=device, dtype=torch.bfloat16),
        )

    def forward(
        self,
        fisheye: Tensor,
        carry: Optional[HRMEdgeCarry] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, HRMEdgeCarry]:
        """
        Forward pass through hierarchical reasoning.

        Args:
            fisheye: (batch, x_pixels) per-edge fisheye features
            carry: optional carry state from previous step

        Returns:
            fraction, ptt, stop: (batch,) each
            new_carry: updated carry state
        """
        batch = fisheye.shape[0]
        device = fisheye.device

        # Input embedding
        input_emb = self.embed(fisheye)  # (batch, 1, hidden_size)

        # Initialize carry
        if carry is None:
            carry = self.init_carry(batch, device)

        cos_sin = self.rotary_emb()

        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _ in range(self.H_cycles):
                for _ in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_emb, cos_sin)
                z_H = self.H_level(z_H, z_L, cos_sin)

        # Use final H-level representation for prediction
        z = z_H.squeeze(1)  # (batch, hidden_size)

        fraction = torch.sigmoid(self.fraction_head(z)).squeeze(-1)
        ptt = torch.sigmoid(self.ptt_head(z)).squeeze(-1)
        stop = torch.sigmoid(self.stop_head(z)).squeeze(-1)

        new_carry = HRMEdgeCarry(z_H=z_H.detach(), z_L=z_L.detach())
        return fraction, ptt, stop, new_carry

    # ── Rotation growth ────────────────────────────────────────────────────────

    def _make_quadrant_block(self, blocks: List[Tensor]) -> Tensor:
        """Stack 4 rotated blocks as 2×2 quadrant matrix."""
        top = torch.cat([blocks[0], blocks[1]], dim=0)
        bot = torch.cat([blocks[2], blocks[3]], dim=0)
        return torch.cat([top, bot], dim=1)

    def grow_hidden_size(self, new_hidden_size: int):
        """
        Grow hidden_size by 4× using rotational expansion.

        Quadrant layout preserved forever in top-left:
            ┌────────┬────────┐
            │  0°    │ 180°   │  ← original, always preserved
            ├────────┼────────┤
            │  90°   │ 270°   │
            └────────┴────────┘
        """
        old_hs = self.hidden_size
        factor = new_hidden_size // old_hs
        if factor != 4:
            raise ValueError(f"Can only grow by 4×, got {factor}×")

        sd = self.state_dict()
        new_sd = {}

        for name, param in sd.items():
            if 'H_init' in name or 'L_init' in name:
                if param.dim() == 1:
                    new_sd[name] = torch.cat([param]*4, dim=0)
                else:
                    new_sd[name] = param
            elif 'embed.proj.weight' in name:
                W = param
                rotated = [W, rotate_180(W), rotate_90(W), rotate_270(W)]
                new_W = torch.cat(rotated, dim=0)
                new_sd[name] = new_W
            elif 'embed.proj.bias' in name:
                new_sd[name] = torch.cat([param]*4, dim=0)
            elif 'fraction_head' in name or 'ptt_head' in name or 'stop_head' in name:
                if 'weight' in name:
                    new_sd[name] = nn.init.zeros_(torch.empty(new_hidden_size, new_hidden_size))
                else:
                    new_sd[name] = nn.init.zeros_(torch.empty(new_hidden_size))
            elif 'H_level.layers' in name or 'L_level.layers' in name:
                new_sd[name] = self._expand_param(name, param, old_hs, new_hidden_size)
            else:
                new_sd[name] = param

        self.hidden_size = new_hidden_size
        self.num_heads = max(1, new_hidden_size // 64)
        self.load_state_dict(new_sd, strict=False)

        for head in [self.fraction_head, self.ptt_head, self.stop_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _expand_param(self, name: str, param: Tensor, old_hs: int, new_hs: int) -> Tensor:
        """Expand a parameter by 4× in the hidden dimension."""
        if 'q_proj.weight' in name or 'k_proj.weight' in name or 'v_proj.weight' in name:
            rotated = [param, rotate_180(param), rotate_90(param), rotate_270(param)]
            return torch.cat(rotated, dim=0)
        elif 'o_proj.weight' in name:
            rotated = [param, rotate_180(param), rotate_90(param), rotate_270(param)]
            top = torch.cat([rotated[0], rotated[1]], dim=0)
            bot = torch.cat([rotated[2], rotated[3]], dim=0)
            return torch.cat([top, bot], dim=1)
        elif 'gate_up_proj.weight' in name:
            rotated = [param, rotate_180(param), rotate_90(param), rotate_270(param)]
            return torch.cat(rotated, dim=1)
        elif 'down_proj.weight' in name:
            rotated = [param, rotate_180(param), rotate_90(param), rotate_270(param)]
            return torch.cat(rotated, dim=0)
        elif 'bias' in name:
            return torch.cat([param]*4, dim=0)
        return param

    def grow_layers(self, level: str = 'H'):
        """
        Grow layers by 4× using rotational expansion (NOT single layer addition).

        Creates 3 additional rotated copies of each existing layer:
        - 0°: original (preserved)
        - 180°: rotated copy
        - 90°: rotated copy
        - 270°: rotated copy

        Args:
            level: 'H' or 'L' - which reasoning module to deepen
        """
        assert level in ('H', 'L')
        old_layers = self.H_layers if level == 'H' else self.L_layers

        module = getattr(self, f'{level}_level')
        new_layers_list = []

        for layer in module.layers:
            src_sd = layer.state_dict()

            for rotation, rot_fn in [(0, None), (180, rotate_180), (90, rotate_90), (270, rotate_270)]:
                new_layer = HRMBlock(self.hidden_size, self.num_heads, self.expansion)
                if rotation == 0:
                    new_layer.load_state_dict(src_sd)
                else:
                    new_sd = {}
                    for name, param in src_sd.items():
                        if 'weight' in name:
                            new_sd[name] = rot_fn(param)
                        else:
                            new_sd[name] = param
                    new_layer.load_state_dict(new_sd)
                new_layers_list.append(new_layer)

        module.layers = nn.ModuleList(new_layers_list)

        if level == 'H':
            self.H_layers = old_layers * 4
        else:
            self.L_layers = old_layers * 4

    def get_square_cube_size(self) -> int:
        """Return current square cube size (hidden_size is the reference)."""
        return self.hidden_size


# ── AccelModelHRM wrapper ──────────────────────────────────────────────────────

class HierarchicalReasoningModel:
    """Hierarchical Reasoning Model backed by HRMEdgePredictor."""
    def __init__(
        self,
        n_edges: int = 0,
        learning_rate: float = 0.001,
        y_depth: int = 200,
        x_pixels: int = 20,
        curvature: float = 2.0,
        h_dim: int = 4,
        z_dim: int = 4,
        prediction_depth: int = 1,
        H_layers: int = 2,
        L_layers: int = 2,
        H_cycles: int = 2,
        L_cycles: int = 2,
        **kwargs,
    ):
        self.y_depth = y_depth
        self.x_pixels = x_pixels
        self._lr = learning_rate
        self.curvature = curvature
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.prediction_depth = prediction_depth
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.edge_names: List[Tuple[str, str]] = []
        self.node_names: List[str] = []
        self.node_to_idx: Dict[str, int] = {}

        self._model: Optional[HRMEdgePredictor] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

        self._close_buffer: Dict[Tuple[str, str], List[float]] = {}
        self._max_history = max(y_depth + 100, 500)
        self._prediction_queue: Dict[Tuple[str, str], List] = {}
        self._carry: Dict[Tuple[str, str], HRMEdgeCarry] = {}

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

        self._model = HRMEdgePredictor(
            x_pixels=self.x_pixels,
            hidden_size=self.h_dim,
            num_heads=max(1, self.h_dim // 64),
            H_layers=self.H_layers,
            L_layers=self.L_layers,
            H_cycles=self.H_cycles,
            L_cycles=self.L_cycles,
        )
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        for edge in edges:
            self._prediction_queue[edge] = []
            self._carry[edge] = None

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
        import numpy as np
        for edge in self.edge_names:
            if edge in graph.edges:
                df = graph.edges[edge]
                if bar_idx < len(df):
                    close = float(df['close'].iloc[bar_idx])
                    buf = self._close_buffer.get(edge, [])
                    buf.append(close)
                    if len(buf) > self._max_history:
                        buf = buf[-self._max_history:]
                    self._close_buffer[edge] = buf

    def predict(self, graph, bar_idx: int = -1) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        if not self.edge_names or self._model is None:
            return {}

        predictions = {}
        with torch.no_grad():
            for edge in self.edge_names:
                fisheye = torch.tensor(self._get_fisheye(edge), dtype=torch.float32).unsqueeze(0)  # (1, x_pixels)
                fraction, ptt, stop, carry = self._model(fisheye, self._carry.get(edge))
                f, p, s = float(fraction), float(ptt), float(stop)
                predictions[edge] = (f, p, s)
                self._carry[edge] = carry
                self._prediction_queue[edge].append({'fisheye': self._get_fisheye(edge), 'bar_idx': bar_idx})
                if len(self._prediction_queue[edge]) > self.prediction_depth:
                    self._prediction_queue[edge] = self._prediction_queue[edge][-self.prediction_depth:]

        return predictions

    def update(self, graph, actual_accels: Dict[Tuple[str, str], float], bar_idx: int = -1,
               actual_velocities=None, hit_ptt=None, hit_stop=None) -> Optional[float]:
        if not self.edge_names or self._model is None or self._optimizer is None:
            return None

        self.update_prices(graph, bar_idx)

        if hit_ptt is None or hit_stop is None:
            return None

        self._optimizer.zero_grad()
        import numpy as np

        total_loss = torch.tensor(0.0)
        n_scored = 0

        for edge in self.edge_names:
            queue = self._prediction_queue.get(edge, [])
            if len(queue) < self.prediction_depth:
                continue

            frame = queue[0]

            ptt_target = 1.0 if hit_ptt.get(edge, False) else 0.0
            stop_target = 1.0 if hit_stop.get(edge, False) else 0.0

            if hit_ptt.get(edge, False):
                frac_target = 1.0
            elif hit_stop.get(edge, False):
                frac_target = 0.0
            else:
                frac_target = 0.5

            fisheye = torch.tensor(frame['fisheye'], dtype=torch.float32).unsqueeze(0)  # (1, x_pixels)
            pred_frac, pred_ptt, pred_stop, _ = self._model(fisheye)

            frac_loss = F.binary_cross_entropy(pred_frac, torch.tensor([frac_target]))
            ptt_loss = F.binary_cross_entropy(pred_ptt, torch.tensor([ptt_target]))
            stop_loss = F.binary_cross_entropy(pred_stop, torch.tensor([stop_target]))

            total_loss = total_loss + frac_loss + ptt_loss + stop_loss
            n_scored += 1

        if n_scored > 0:
            loss = total_loss / n_scored
            loss.backward()
            self._optimizer.step()
            return loss.item()
        return 0.0

    def high_level_plan(self, graph) -> Dict[str, str]:
        return {}

    def get_hrm_stats(self) -> List[Dict]:
        return []

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
            'H_layers': self.H_layers,
            'L_layers': self.L_layers,
            'H_cycles': self.H_cycles,
            'L_cycles': self.L_cycles,
        }
        torch.save(state, path)

    def load(self, path: str = "model_weights.pt"):
        if not Path(path).exists():
            return
        state = None
        try:
            state = torch.load(path, weights_only=True)
        except Exception:
            try:
                state = torch.load(path, weights_only=False)
            except Exception as exc:
                print(f"Skipping checkpoint load for {path}: unsupported legacy format ({exc})")
                return
        if self._model is not None and 'model' in state:
            self._model.load_state_dict(state['model'])

    def grow(self, dim: str):
        """Grow one dimension by 4× using rotational expansion.
        
        dim: 'h' for hidden_size, 'H' or 'L' for layers
        """
        if self._model is None:
            return

        if dim == 'h':
            current = self._model.hidden_size
            idx = SQUARE_CUBE_SIZES.index(current) if current in SQUARE_CUBE_SIZES else -1
            if idx >= 0 and idx + 1 < len(SQUARE_CUBE_SIZES):
                new_size = SQUARE_CUBE_SIZES[idx + 1]
            elif idx < 0:
                new_size = current * 4
            else:
                new_size = current * 4
            self._model.grow_hidden_size(new_size)
            self.h_dim = self._model.hidden_size
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        elif dim in ('H', 'L'):
            self._model.grow_layers(dim)
            if dim == 'H':
                self.H_layers = self._model.H_layers
            else:
                self.L_layers = self._model.L_layers
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)


# ── Backwards compat ──────────────────────────────────────────────────────────

# fisheye functions defined inline above
def fisheye_boundaries(y_depth: int, x_pixels: int, curvature: float) -> List[int]:
    """Fisheye bucket boundaries."""
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

def fisheye_sample(candles, boundaries: List[int]) -> List[float]:
    """Sample candles into fisheye buckets."""
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
