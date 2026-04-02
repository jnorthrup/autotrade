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

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bag_io import build_bag_model_frames as build_bag_model_frames_df
from bag_io import bind_spend_budgets

try:
    import sympy  # noqa: F401
except Exception:
    sympy = None

# Keep graph_showdown self-contained instead of reaching into the nested HRM repo.
from layers_fallback import rms_norm, SwiGLU, Attention, RotaryEmbedding


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve the requested runtime device."""
    choice = (preferred or "auto").lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        raise RuntimeError("Requested device=mps but MPS is not available")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested device=cuda but CUDA is not available")
    if choice != "auto":
        raise ValueError(f"Unsupported device preference: {preferred}")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _checkpoint_cas(metadata: Dict[str, object], state_dict: Dict[str, Tensor]) -> str:
    """Hash checkpoint content into a stable content-addressable signature."""
    hasher = hashlib.sha256()
    hasher.update(CHECKPOINT_CAS_VERSION.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )
    hasher.update(b"\0")

    for name in sorted(state_dict.keys()):
        tensor = state_dict[name].detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(tensor.numpy().tobytes())
        hasher.update(b"\0")

    return hasher.hexdigest()


# ── Square cube sizes ──────────────────────────────────────────────────────────

SQUARE_CUBE_SIZES = [4, 16, 64, 256]  # hidden_size progression
CHECKPOINT_CAS_VERSION = "hrm-checkpoint-v1"


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
        num_nodes: int = 1,
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
        self.num_nodes = max(1, int(num_nodes))
        self.num_heads = num_heads
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.expansion = expansion

        # Input embedding from fisheye features
        self.embed = FisheyeEmbedding(x_pixels, hidden_size)
        self.base_node_embed = nn.Embedding(self.num_nodes, hidden_size)
        self.quote_node_embed = nn.Embedding(self.num_nodes, hidden_size)
        self.pair_index_proj = nn.Linear(4, hidden_size)
        nn.init.zeros_(self.base_node_embed.weight)
        nn.init.zeros_(self.quote_node_embed.weight)
        nn.init.zeros_(self.pair_index_proj.weight)
        nn.init.zeros_(self.pair_index_proj.bias)

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
            trunc_normal_init_(torch.empty(hidden_size, dtype=torch.float32), std=1),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(hidden_size, dtype=torch.float32), std=1),
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

    def resize_node_embeddings(self, new_num_nodes: int):
        """Resize bag-index vocab without rebuilding the whole predictor."""
        new_num_nodes = max(1, int(new_num_nodes))
        if new_num_nodes == self.num_nodes:
            return

        base = nn.Embedding(new_num_nodes, self.hidden_size).to(self.base_node_embed.weight.device)
        quote = nn.Embedding(new_num_nodes, self.hidden_size).to(self.quote_node_embed.weight.device)
        nn.init.zeros_(base.weight)
        nn.init.zeros_(quote.weight)

        rows_to_copy = min(self.num_nodes, new_num_nodes)
        with torch.no_grad():
            base.weight[:rows_to_copy] = self.base_node_embed.weight[:rows_to_copy]
            quote.weight[:rows_to_copy] = self.quote_node_embed.weight[:rows_to_copy]

        self.base_node_embed = base
        self.quote_node_embed = quote
        self.num_nodes = new_num_nodes

    def _pair_input_embedding(self, base_idx: Tensor, quote_idx: Tensor) -> Tensor:
        base_idx = base_idx.long()
        quote_idx = quote_idx.long()
        base_emb = self.base_node_embed(base_idx)
        quote_emb = self.quote_node_embed(quote_idx)

        denom = float(max(1, self.num_nodes - 1))
        base_norm = base_idx.to(dtype=base_emb.dtype) / denom
        quote_norm = quote_idx.to(dtype=base_emb.dtype) / denom
        pair_scalars = torch.stack(
            [
                base_norm,
                quote_norm,
                base_norm - quote_norm,
                base_norm * quote_norm,
            ],
            dim=-1,
        )
        return (base_emb + quote_emb + self.pair_index_proj(pair_scalars)).unsqueeze(1)

    def init_carry(self, batch_size: int, device: torch.device) -> HRMEdgeCarry:
        """Initialize empty carry states."""
        return HRMEdgeCarry(
            z_H=self.H_init.view(1, 1, self.hidden_size).expand(batch_size, -1, -1).clone().to(device=device),
            z_L=self.L_init.view(1, 1, self.hidden_size).expand(batch_size, -1, -1).clone().to(device=device),
        )

    def forward(
        self,
        fisheye: Tensor,
        base_idx: Optional[Tensor] = None,
        quote_idx: Optional[Tensor] = None,
        carry: Optional[HRMEdgeCarry] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, HRMEdgeCarry]:
        """
        Forward pass through hierarchical reasoning.

        Args:
            fisheye: (batch, x_pixels) per-edge fisheye features
            base_idx: (batch,) ordered base-coin bag index
            quote_idx: (batch,) ordered quote-coin bag index
            carry: optional carry state from previous step

        Returns:
            fraction, ptt, stop: (batch,) each
            new_carry: updated carry state
        """
        batch = fisheye.shape[0]
        device = fisheye.device

        # Input embedding
        input_emb = self.embed(fisheye)  # (batch, 1, hidden_size)
        if base_idx is not None or quote_idx is not None:
            if base_idx is None or quote_idx is None:
                raise ValueError("base_idx and quote_idx must be provided together")
            input_emb = input_emb + self._pair_input_embedding(base_idx, quote_idx)

        # Initialize carry
        if carry is None:
            carry = self.init_carry(batch, device)

        cos_sin = self.rotary_emb()

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

    @staticmethod
    def _rotate_same_shape(param: Tensor, rotation: int) -> Tensor:
        if rotation == 0:
            return param.detach().clone()
        if param.dim() == 1:
            return torch.flip(param, dims=(0,))
        if param.dim() < 2:
            return param.detach().clone()
        if param.shape[-2] == param.shape[-1]:
            turns = {90: 1, 180: 2, 270: 3}[rotation]
            return torch.rot90(param, k=turns, dims=(-2, -1))
        if rotation == 180:
            return torch.flip(param, dims=(-2, -1))
        if rotation == 90:
            return torch.flip(param, dims=(-2,))
        if rotation == 270:
            return torch.flip(param, dims=(-1,))
        raise ValueError(f"Unsupported rotation: {rotation}")

    @classmethod
    def _growth_variants(cls, param: Tensor) -> List[Tensor]:
        if param.dim() >= 2 and param.shape[-2] == param.shape[-1]:
            return [
                cls._rotate_same_shape(param, 0),
                cls._rotate_same_shape(param, 180),
                cls._rotate_same_shape(param, 90),
                cls._rotate_same_shape(param, 270),
            ]
        rot180 = cls._rotate_same_shape(param, 180)
        # Axis-only fallback uses 0,180,180,0 to preserve the original block.
        return [
            cls._rotate_same_shape(param, 0),
            rot180,
            rot180.detach().clone(),
            cls._rotate_same_shape(param, 0),
        ]

    @classmethod
    def _expand_axis_to(cls, param: Tensor, axis: int, target_size: int) -> Tensor:
        current_size = param.shape[axis]
        if current_size == target_size:
            return param.detach().clone()
        if current_size > target_size:
            slicer = [slice(None)] * param.dim()
            slicer[axis] = slice(0, target_size)
            return param[tuple(slicer)].detach().clone()

        variants = cls._growth_variants(param)
        pieces: List[Tensor] = []
        remaining = target_size
        variant_idx = 0
        while remaining > 0:
            block = variants[variant_idx % len(variants)]
            block_size = block.shape[axis]
            if block_size <= remaining:
                pieces.append(block.detach().clone())
                remaining -= block_size
            else:
                slicer = [slice(None)] * block.dim()
                slicer[axis] = slice(0, remaining)
                pieces.append(block[tuple(slicer)].detach().clone())
                remaining = 0
            variant_idx += 1
        return torch.cat(pieces, dim=axis)

    @classmethod
    def _expand_param_to_shape(cls, param: Tensor, target_shape: torch.Size) -> Tensor:
        if tuple(param.shape) == tuple(target_shape):
            return param.detach().clone()
        if param.dim() != len(target_shape):
            raise ValueError(f"Cannot expand {tuple(param.shape)} into {tuple(target_shape)}")

        expanded = param.detach().clone()
        if expanded.dim() == 1:
            return cls._expand_axis_to(expanded, 0, target_shape[0])

        if expanded.shape[1] != target_shape[1]:
            expanded = cls._expand_axis_to(expanded, 1, target_shape[1])
        if expanded.shape[0] != target_shape[0]:
            expanded = cls._expand_axis_to(expanded, 0, target_shape[0])
        if tuple(expanded.shape) != tuple(target_shape):
            slicer = tuple(slice(0, size) for size in target_shape)
            expanded = expanded[slicer].detach().clone()
        return expanded

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
        device = next(self.parameters()).device
        grown = HRMEdgePredictor(
            x_pixels=self.x_pixels,
            hidden_size=new_hidden_size,
            num_nodes=self.num_nodes,
            num_heads=max(1, new_hidden_size // 64),
            H_layers=self.H_layers,
            L_layers=self.L_layers,
            H_cycles=self.H_cycles,
            L_cycles=self.L_cycles,
            expansion=self.expansion,
        ).to(device)

        old_sd = self.state_dict()
        target_sd = grown.state_dict()
        expanded_sd = {}
        for name, target in target_sd.items():
            source = old_sd.get(name)
            if source is None:
                expanded_sd[name] = target
                continue
            expanded = self._expand_param_to_shape(source, target.shape).to(dtype=target.dtype, device=target.device)
            expanded_sd[name] = expanded

        grown.load_state_dict(expanded_sd, strict=False)
        return grown

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
                        new_sd[name] = self._rotate_same_shape(param, rotation)
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
        device: str = "auto",
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
        self.device_preference = device

        self.edge_names: List[Tuple[str, ...]] = []
        self.node_names: List[str] = []
        self.node_to_idx: Dict[str, int] = {}
        self.num_nodes = 0

        self._model: Optional[HRMEdgePredictor] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device: torch.device = get_device(self.device_preference)

        self._close_buffer: Dict[Tuple[str, ...], List[float]] = {}
        self._max_history = max(y_depth + 100, 500)
        self._prediction_queue: Dict[Tuple[str, ...], List] = {}
        self._carry: Dict[Tuple[str, ...], HRMEdgeCarry] = {}
        self._edge_observation_count: Dict[Tuple[str, ...], int] = {}
        self._observed_edges_for_bar: List[Tuple[str, ...]] = []
        self._last_price_bar_idx: Optional[int] = None
        self._fisheye_boundaries = fisheye_boundaries(self.y_depth, self.x_pixels, self.curvature)
        self._profile_enabled = False
        self._profile_stats: Dict[str, float] = {}
        self.reset_profile_stats()

    def _checkpoint_metadata(self) -> Dict[str, object]:
        return {
            'model_class': 'HRMEdgePredictor',
            'x_pixels': self.x_pixels,
            'y_depth': self.y_depth,
            'curvature': self.curvature,
            'h_dim': self.h_dim,
            'z_dim': self.z_dim,
            'num_nodes': int(self.num_nodes or 0),
            'prediction_depth': self.prediction_depth,
            'H_layers': self.H_layers,
            'L_layers': self.L_layers,
            'H_cycles': self.H_cycles,
            'L_cycles': self.L_cycles,
        }

    def model_cas_signature(self) -> Optional[str]:
        if self._model is None:
            return None
        return _checkpoint_cas(self._checkpoint_metadata(), self._model.state_dict())

    def _reset_edge_runtime_state(self):
        self._close_buffer = {}
        self._prediction_queue = {}
        self._carry = {}
        self._edge_observation_count = {}
        self._observed_edges_for_bar = []
        self._last_price_bar_idx = None
        for edge in self.edge_names:
            self._close_buffer[edge] = []
            self._prediction_queue[edge] = []
            self._carry[edge] = None
            self._edge_observation_count[edge] = 0

    def _build_model(self):
        self._device = get_device(self.device_preference)
        self._model = HRMEdgePredictor(
            x_pixels=self.x_pixels,
            hidden_size=self.h_dim,
            num_nodes=max(1, self.num_nodes or len(self.node_names) or 1),
            num_heads=max(1, self.h_dim // 64),
            H_layers=self.H_layers,
            L_layers=self.L_layers,
            H_cycles=self.H_cycles,
            L_cycles=self.L_cycles,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        self._max_history = max(self.y_depth + 100, 500)
        self._reset_edge_runtime_state()

    @staticmethod
    def _count_layers(state_dict: Dict[str, Tensor], prefix: str) -> int:
        token = f"{prefix}.layers."
        layer_indices = set()
        for name in state_dict:
            if not name.startswith(token):
                continue
            remainder = name[len(token):]
            layer_idx, _, _ = remainder.partition(".")
            if layer_idx.isdigit():
                layer_indices.add(int(layer_idx))
        return (max(layer_indices) + 1) if layer_indices else 0

    @classmethod
    def _checkpoint_config(cls, state: Dict) -> Dict[str, float]:
        model_state = state.get('model', {}) if isinstance(state, dict) else {}

        h_dim = state.get('h_dim')
        if h_dim is None and 'H_init' in model_state:
            h_dim = int(model_state['H_init'].shape[0])

        z_dim = state.get('z_dim', h_dim)

        x_pixels = state.get('x_pixels')
        if x_pixels is None and 'embed.proj.weight' in model_state:
            x_pixels = int(model_state['embed.proj.weight'].shape[1])

        num_nodes = state.get('num_nodes')
        if num_nodes is None and 'base_node_embed.weight' in model_state:
            num_nodes = int(model_state['base_node_embed.weight'].shape[0])

        return {
            'x_pixels': int(x_pixels) if x_pixels is not None else None,
            'y_depth': int(state['y_depth']) if 'y_depth' in state else None,
            'curvature': float(state['curvature']) if 'curvature' in state else None,
            'h_dim': int(h_dim) if h_dim is not None else None,
            'z_dim': int(z_dim) if z_dim is not None else None,
            'num_nodes': int(num_nodes) if num_nodes is not None else None,
            'prediction_depth': int(state['prediction_depth']) if 'prediction_depth' in state else None,
            'H_layers': int(state['H_layers']) if 'H_layers' in state else cls._count_layers(model_state, 'H_level'),
            'L_layers': int(state['L_layers']) if 'L_layers' in state else cls._count_layers(model_state, 'L_level'),
            'H_cycles': int(state['H_cycles']) if 'H_cycles' in state else None,
            'L_cycles': int(state['L_cycles']) if 'L_cycles' in state else None,
        }

    def _apply_checkpoint_config(self, state: Dict):
        for key, value in self._checkpoint_config(state).items():
            if value is None:
                continue
            setattr(self, key, value)
        self._max_history = max(self.y_depth + 100, 500)
        self._fisheye_boundaries = fisheye_boundaries(self.y_depth, self.x_pixels, self.curvature)

    def register_edges(self, edges: List[Tuple[str, ...]]):
        self.edge_names = edges

        node_set = set()
        for edge in edges:
            if len(edge) == 2:
                base, quote = edge
            elif len(edge) == 3:
                _, base, quote = edge
            else:
                raise ValueError(f"Unsupported edge shape: {edge!r}")
            node_set.add(base)
            node_set.add(quote)

        if "USD" in node_set:
            node_set.discard("USD")
            self.node_names = ["USD"] + sorted(node_set)
        else:
            self.node_names = sorted(node_set)

        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}
        self.num_nodes = len(self.node_names)
        if self._model is None:
            self._build_model()
        else:
            self._model.resize_node_embeddings(max(1, self.num_nodes))
            self._reset_edge_runtime_state()

    def _edge_node_indices(self, edge: Tuple[str, ...]) -> Tuple[int, int]:
        if len(edge) == 2:
            base, quote = edge
        elif len(edge) == 3:
            _, base, quote = edge
        else:
            raise ValueError(f"Unsupported edge shape: {edge!r}")
        return self.node_to_idx[str(base)], self.node_to_idx[str(quote)]

    def _edge_index_batch(self, edges: List[Tuple[str, ...]]) -> Tuple[Tensor, Tensor]:
        base_idx, quote_idx = zip(*(self._edge_node_indices(edge) for edge in edges))
        return (
            torch.as_tensor(np.asarray(base_idx, dtype=np.int64), device=self._device),
            torch.as_tensor(np.asarray(quote_idx, dtype=np.int64), device=self._device),
        )

    def build_bag_model_frames(
        self,
        graph,
        bar_idx: int,
        *,
        value_asset: str = "USD",
        free_qty=None,
        reserved_qty=None,
        route_discount=None,
    ):
        if not self.edge_names:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        fisheye_map = {edge: self._get_fisheye(edge) for edge in self.edge_names}
        carry_map = {edge: self._carry.get(edge) for edge in self.edge_names}
        return build_bag_model_frames_df(
            graph,
            bar_idx=bar_idx,
            edge_names=self.edge_names,
            node_to_idx=self.node_to_idx,
            value_asset=value_asset,
            edge_fisheyes=fisheye_map,
            edge_carries=carry_map,
            free_qty=free_qty,
            reserved_qty=reserved_qty,
            route_discount=route_discount,
        )

    @staticmethod
    def bind_spend_budgets(pred_df: pd.DataFrame, node_state_df: pd.DataFrame) -> pd.DataFrame:
        return bind_spend_budgets(pred_df, node_state_df)

    def _get_fisheye(self, edge: Tuple[str, ...]) -> List[float]:
        closes = self._close_buffer.get(edge, [])
        if len(closes) < self.y_depth:
            return [0.0] * self.x_pixels
        closes_array = np.asarray(closes[-self.y_depth:], dtype=np.float32)
        values = fisheye_sample(closes_array, self._fisheye_boundaries)
        values = values[:self.x_pixels]
        while len(values) < self.x_pixels:
            values.append(0.0)
        return values

    def reset_profile_stats(self):
        self._profile_stats = {
            'bars_observed': 0.0,
            'stale_frames_dropped': 0.0,
            'update_prices_seconds': 0.0,
            'predict_batches': 0.0,
            'predict_edges': 0.0,
            'predict_prepare_seconds': 0.0,
            'predict_forward_seconds': 0.0,
            'update_batches': 0.0,
            'update_edges': 0.0,
            'update_prepare_seconds': 0.0,
            'update_forward_backward_seconds': 0.0,
        }

    def set_profile_enabled(self, enabled: bool):
        self._profile_enabled = enabled
        self.reset_profile_stats()

    def get_profile_stats(self) -> Dict[str, float]:
        stats = dict(self._profile_stats)
        stats['device_type'] = self._device.type
        return stats

    def _record_profile(self, key: str, value: float):
        if self._profile_enabled:
            self._profile_stats[key] += value

    def _sync_device_for_profile(self):
        if not self._profile_enabled:
            return
        if self._device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        elif self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)

    def _edge_has_full_history(self, edge: Tuple[str, str]) -> bool:
        return len(self._close_buffer.get(edge, [])) >= self.y_depth

    def _prediction_ready_edges(self, bar_idx: Optional[int] = None) -> List[Tuple[str, str]]:
        if bar_idx is not None and self._last_price_bar_idx != bar_idx:
            return []
        return [edge for edge in self._observed_edges_for_bar if self._edge_has_full_history(edge)]

    def ready_for_prediction(self, bar_idx: Optional[int] = None) -> bool:
        return bool(self._prediction_ready_edges(bar_idx))

    def _drop_stale_predictions(self, edge: Tuple[str, str]):
        queue = self._prediction_queue.get(edge, [])
        observed = self._edge_observation_count.get(edge, 0)
        dropped = 0
        while queue and observed - queue[0]['observation_count'] > self.prediction_depth:
            queue.pop(0)
            dropped += 1
        if dropped:
            self._record_profile('stale_frames_dropped', float(dropped))

    def _matured_prediction_frame(self, edge: Tuple[str, str]) -> Optional[Dict]:
        self._drop_stale_predictions(edge)
        queue = self._prediction_queue.get(edge, [])
        if not queue:
            return None
        observed = self._edge_observation_count.get(edge, 0)
        frame = queue[0]
        if observed - frame['observation_count'] == self.prediction_depth:
            return frame
        return None

    def ready_for_update(
        self,
        bar_idx: Optional[int] = None,
        actual_accels: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> bool:
        if bar_idx is not None and self._last_price_bar_idx != bar_idx:
            return False
        candidate_edges = actual_accels.keys() if actual_accels is not None else self._observed_edges_for_bar
        return any(self._matured_prediction_frame(edge) is not None for edge in candidate_edges)

    @staticmethod
    def _slice_carry(carry: HRMEdgeCarry, idx: int) -> HRMEdgeCarry:
        return HRMEdgeCarry(
            z_H=carry.z_H[idx:idx + 1].detach().clone(),
            z_L=carry.z_L[idx:idx + 1].detach().clone(),
        )

    @staticmethod
    def _concat_carries(carries: Iterable[HRMEdgeCarry]) -> HRMEdgeCarry:
        carry_list = list(carries)
        return HRMEdgeCarry(
            z_H=torch.cat([carry.z_H for carry in carry_list], dim=0),
            z_L=torch.cat([carry.z_L for carry in carry_list], dim=0),
        )

    def _build_batch_carry(self, edges: List[Tuple[str, str]]) -> HRMEdgeCarry:
        carry = self._model.init_carry(len(edges), self._device)
        for idx, edge in enumerate(edges):
            edge_carry = self._carry.get(edge)
            if edge_carry is None:
                continue
            carry.z_H[idx:idx + 1] = edge_carry.z_H
            carry.z_L[idx:idx + 1] = edge_carry.z_L
        return carry

    def update_prices(self, graph, bar_idx: int) -> List[Tuple[str, str]]:
        start = time.perf_counter() if self._profile_enabled else None
        if bar_idx < 0 or bar_idx >= len(graph.common_timestamps):
            self._last_price_bar_idx = None
            self._observed_edges_for_bar = []
            return []
        ts = graph.common_timestamps[bar_idx]
        observed_edges = []
        for edge in self.edge_names:
            df = graph.edges.get(edge)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            if hasattr(graph, "edge_price_components"):
                close = float(graph.edge_price_components(edge, row)["close"])
            else:
                close_value = row.get("close", 0.0)
                close = float(close_value)
                if getattr(graph, "edge_is_inverted", {}).get(edge, False) and close > 0:
                    close = 1.0 / close
            buf = self._close_buffer[edge]
            buf.append(close)
            if len(buf) > self._max_history:
                del buf[:-self._max_history]
            self._edge_observation_count[edge] += 1
            observed_edges.append(edge)
        self._last_price_bar_idx = bar_idx
        self._observed_edges_for_bar = observed_edges
        self._record_profile('bars_observed', 1.0)
        if start is not None:
            self._record_profile('update_prices_seconds', time.perf_counter() - start)
        return observed_edges

    def predict(self, graph, bar_idx: int = -1) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        if not self.edge_names or self._model is None:
            return {}

        ready_edges = self._prediction_ready_edges(bar_idx)
        if not ready_edges:
            return {}

        prep_start = time.perf_counter() if self._profile_enabled else None
        fisheye_rows = [self._get_fisheye(edge) for edge in ready_edges]
        fisheye_batch = torch.as_tensor(np.asarray(fisheye_rows, dtype=np.float32), device=self._device)
        base_idx_batch, quote_idx_batch = self._edge_index_batch(ready_edges)
        input_carry = self._build_batch_carry(ready_edges)
        if prep_start is not None:
            self._record_profile('predict_prepare_seconds', time.perf_counter() - prep_start)

        self._sync_device_for_profile()
        fwd_start = time.perf_counter() if self._profile_enabled else None
        with torch.no_grad():
            fraction, ptt, stop, output_carry = self._model(
                fisheye_batch,
                base_idx_batch,
                quote_idx_batch,
                input_carry,
            )
        self._sync_device_for_profile()
        if fwd_start is not None:
            self._record_profile('predict_forward_seconds', time.perf_counter() - fwd_start)
        self._record_profile('predict_batches', 1.0)
        self._record_profile('predict_edges', float(len(ready_edges)))

        fractions = fraction.detach().cpu().tolist()
        ptts = ptt.detach().cpu().tolist()
        stops = stop.detach().cpu().tolist()
        predictions = {}
        for idx, edge in enumerate(ready_edges):
            predictions[edge] = (float(fractions[idx]), float(ptts[idx]), float(stops[idx]))
            self._carry[edge] = self._slice_carry(output_carry, idx)
            self._prediction_queue[edge].append({
                'fisheye': fisheye_rows[idx],
                'bar_idx': bar_idx,
                'observation_count': self._edge_observation_count[edge],
                'carry': self._slice_carry(input_carry, idx),
            })
        return predictions

    def update(self, graph, actual_accels: Dict[Tuple[str, str], float], bar_idx: int = -1,
               actual_velocities=None, hit_ptt=None, hit_stop=None) -> Optional[float]:
        if not self.edge_names or self._model is None or self._optimizer is None:
            return None

        if hit_ptt is None or hit_stop is None:
            return None
        if bar_idx >= 0 and self._last_price_bar_idx != bar_idx:
            return None

        prep_start = time.perf_counter() if self._profile_enabled else None
        matured_edges: List[Tuple[str, str]] = []
        fisheye_rows: List[List[float]] = []
        carry_rows: List[HRMEdgeCarry] = []
        frac_targets: List[float] = []
        ptt_targets: List[float] = []
        stop_targets: List[float] = []

        for edge in actual_accels:
            frame = self._matured_prediction_frame(edge)
            if frame is None:
                continue
            ptt_target = 1.0 if hit_ptt.get(edge, False) else 0.0
            stop_target = 1.0 if hit_stop.get(edge, False) else 0.0

            if hit_ptt.get(edge, False):
                frac_target = 1.0
            elif hit_stop.get(edge, False):
                frac_target = 0.0
            else:
                frac_target = 0.5

            matured_edges.append(edge)
            fisheye_rows.append(frame['fisheye'])
            carry_rows.append(frame['carry'])
            frac_targets.append(frac_target)
            ptt_targets.append(ptt_target)
            stop_targets.append(stop_target)

        if not matured_edges:
            return None

        fisheye_batch = torch.as_tensor(np.asarray(fisheye_rows, dtype=np.float32), device=self._device)
        base_idx_batch, quote_idx_batch = self._edge_index_batch(matured_edges)
        carry_batch = self._concat_carries(carry_rows)
        frac_targets_tensor = torch.as_tensor(np.asarray(frac_targets, dtype=np.float32), device=self._device)
        ptt_targets_tensor = torch.as_tensor(np.asarray(ptt_targets, dtype=np.float32), device=self._device)
        stop_targets_tensor = torch.as_tensor(np.asarray(stop_targets, dtype=np.float32), device=self._device)
        if prep_start is not None:
            self._record_profile('update_prepare_seconds', time.perf_counter() - prep_start)

        self._optimizer.zero_grad(set_to_none=True)
        self._sync_device_for_profile()
        fwd_start = time.perf_counter() if self._profile_enabled else None
        pred_frac, pred_ptt, pred_stop, _ = self._model(
            fisheye_batch,
            base_idx_batch,
            quote_idx_batch,
            carry_batch,
        )
        frac_loss = F.binary_cross_entropy(pred_frac, frac_targets_tensor)
        ptt_loss = F.binary_cross_entropy(pred_ptt, ptt_targets_tensor)
        stop_loss = F.binary_cross_entropy(pred_stop, stop_targets_tensor)
        loss = frac_loss + ptt_loss + stop_loss
        loss.backward()
        self._optimizer.step()
        self._sync_device_for_profile()
        if fwd_start is not None:
            self._record_profile('update_forward_backward_seconds', time.perf_counter() - fwd_start)
        self._record_profile('update_batches', 1.0)
        self._record_profile('update_edges', float(len(matured_edges)))

        for edge in matured_edges:
            self._prediction_queue[edge].pop(0)
        return float(loss.item())

    def score(
        self,
        graph,
        actual_accels: Dict[Tuple[str, str], float],
        bar_idx: int = -1,
        actual_velocities=None,
        hit_ptt=None,
        hit_stop=None,
    ) -> Optional[float]:
        """Score matured predictions without updating weights.

        This is used for walk-forward holdout evaluation after training has
        finished. Runtime queues still advance so the scorer can replay a
        future segment sequentially.
        """
        if not self.edge_names or self._model is None:
            return None

        if hit_ptt is None or hit_stop is None:
            return None
        if bar_idx >= 0 and self._last_price_bar_idx != bar_idx:
            return None

        matured_edges: List[Tuple[str, str]] = []
        fisheye_rows: List[List[float]] = []
        carry_rows: List[HRMEdgeCarry] = []
        frac_targets: List[float] = []
        ptt_targets: List[float] = []
        stop_targets: List[float] = []

        for edge in actual_accels:
            frame = self._matured_prediction_frame(edge)
            if frame is None:
                continue

            ptt_target = 1.0 if hit_ptt.get(edge, False) else 0.0
            stop_target = 1.0 if hit_stop.get(edge, False) else 0.0

            if hit_ptt.get(edge, False):
                frac_target = 1.0
            elif hit_stop.get(edge, False):
                frac_target = 0.0
            else:
                frac_target = 0.5

            matured_edges.append(edge)
            fisheye_rows.append(frame['fisheye'])
            carry_rows.append(frame['carry'])
            frac_targets.append(frac_target)
            ptt_targets.append(ptt_target)
            stop_targets.append(stop_target)

        if not matured_edges:
            return None

        fisheye_batch = torch.as_tensor(np.asarray(fisheye_rows, dtype=np.float32), device=self._device)
        base_idx_batch, quote_idx_batch = self._edge_index_batch(matured_edges)
        carry_batch = self._concat_carries(carry_rows)
        frac_targets_tensor = torch.as_tensor(np.asarray(frac_targets, dtype=np.float32), device=self._device)
        ptt_targets_tensor = torch.as_tensor(np.asarray(ptt_targets, dtype=np.float32), device=self._device)
        stop_targets_tensor = torch.as_tensor(np.asarray(stop_targets, dtype=np.float32), device=self._device)

        with torch.no_grad():
            pred_frac, pred_ptt, pred_stop, _ = self._model(
                fisheye_batch,
                base_idx_batch,
                quote_idx_batch,
                carry_batch,
            )
            frac_loss = F.binary_cross_entropy(pred_frac, frac_targets_tensor)
            ptt_loss = F.binary_cross_entropy(pred_ptt, ptt_targets_tensor)
            stop_loss = F.binary_cross_entropy(pred_stop, stop_targets_tensor)
            loss = frac_loss + ptt_loss + stop_loss

        for edge in matured_edges:
            self._prediction_queue[edge].pop(0)
        return float(loss.item())

    def high_level_plan(self, graph) -> Dict[str, str]:
        return {}

    def get_hrm_stats(self) -> List[Dict]:
        return []

    def save(self, path: str = "model_weights.pt", checkpoint_type: str = "pretrained"):
        if self._model is None:
            return
        from datetime import datetime
        model_cas = self.model_cas_signature()
        state = {
            'model': self._model.state_dict(),
            'x_pixels': self.x_pixels,
            'y_depth': self.y_depth,
            'curvature': self.curvature,
            'h_dim': self.h_dim,
            'z_dim': self.z_dim,
            'num_nodes': int(self.num_nodes or getattr(self._model, 'num_nodes', 0) or 0),
            'prediction_depth': self.prediction_depth,
            'H_layers': self.H_layers,
            'L_layers': self.L_layers,
            'H_cycles': self.H_cycles,
            'L_cycles': self.L_cycles,
            'checkpoint_timestamp': datetime.now().isoformat(),
            'checkpoint_type': checkpoint_type,
            'model_cas': model_cas,
        }
        torch.save(state, path)

    def load(self, path: str = "model_weights.pt"):
        if not Path(path).exists():
            return
        state = None
        try:
            state = torch.load(path, weights_only=True, map_location=self._device)
        except Exception:
            try:
                state = torch.load(path, weights_only=False, map_location=self._device)
            except Exception as exc:
                print(f"Skipping checkpoint load for {path}: unsupported legacy format ({exc})")
                return
        if not isinstance(state, dict) or 'model' not in state:
            print(f"Skipping checkpoint load for {path}: unsupported checkpoint payload")
            return

        self._apply_checkpoint_config(state)

        self._build_model()
        missing, unexpected = self._model.load_state_dict(state['model'], strict=False)
        if missing or unexpected:
            print(
                f"Checkpoint parameter drift for {path}: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        print(
            f"Loaded checkpoint {path}: "
            f"x_pixels={self.x_pixels}, y_depth={self.y_depth}, "
            f"h_dim={self.h_dim}, z_dim={self.z_dim}, "
            f"H_layers={self.H_layers}, L_layers={self.L_layers}, "
            f"prediction_depth={self.prediction_depth}"
        )

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
            self._model = self._model.grow_hidden_size(new_size).to(self._device)
            self.h_dim = self._model.hidden_size
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
            self._reset_edge_runtime_state()
        elif dim in ('H', 'L'):
            self._model.grow_layers(dim)
            if dim == 'H':
                self.H_layers = self._model.H_layers
            else:
                self.L_layers = self._model.L_layers
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
            self._reset_edge_runtime_state()


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
