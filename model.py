import pandas as pd
"""
model.py - Hierarchical Reasoning Model (HRM) with Square Cube Rotational Growth.
Consolidates layers_fallback.py and hrm_model.py.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cache import build_bag_model_frames, bind_spend_budgets

CosSin = Tuple[torch.Tensor, torch.Tensor]
SQUARE_CUBE_SIZES = [4, 16, 64, 256]
CHECKPOINT_CAS_VERSION = "hrm-checkpoint-v1"

def get_device(preferred: str = "auto") -> torch.device:
    choice = (preferred or "auto").lower()
    if choice == "cpu": return torch.device("cpu")
    if choice == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built(): return torch.device("mps")
        raise RuntimeError("Requested device=mps but MPS is not available")
    if choice == "cuda":
        if torch.cuda.is_available(): return torch.device("cuda")
        raise RuntimeError("Requested device=cuda but CUDA is not available")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def _find_multiple(a, b): return (-(a // -b)) * b

def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q, k = q.to(torch.float32), k.to(torch.float32)
    seq_len = q.shape[1]
    cos, sin = cos[:seq_len], sin[:seq_len]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='linear')
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, bias=self.bias)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    def forward(self): return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.q_proj = CastedLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = CastedLinear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = CastedLinear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = CastedLinear(head_dim * num_heads, hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if cos_sin is not None:
            query, key = apply_rotary_pos_emb(query, key, cos_sin[0], cos_sin[1])

        if self.num_key_value_heads != self.num_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(n_rep, dim=2)
            value = value.repeat_interleave(n_rep, dim=2)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_weights.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_output = torch.matmul(F.softmax(attn_weights, dim=-1), value).transpose(1, 2).contiguous()
        return self.o_proj(attn_output.view(batch_size, seq_len, -1))

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)
    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def trunc_normal_init_(tensor, std=1.0): return nn.init.normal_(tensor, mean=0.0, std=std)

class PancakeEmbedding(nn.Module):
    """Pancake: flatten multi-slot features into a single branded vector.
    
    From mp-superproject KlineViewUtil.pancake: N rows x W cols -> 1D N*W vector,
    each slot branded by column/rowIndex. Adds:
    - time-axis cyclical one-hots (DateShed-style: minute-of-hour, hour-of-day, day-of-week)
    - bag slot one-hot (which edges in the bag are active this bar)
    """
    def __init__(self, x_pixels: int, hidden_size: int, n_time_features: int = 18, max_bag_slots: int = 32):
        super().__init__()
        self.x_pixels = x_pixels
        self.max_bag_slots = max_bag_slots
        # Pancake input: fisheye(x_pixels) + time_features(n_time_features) + bag_one_hot(max_bag_slots)
        total_features = x_pixels + n_time_features + max_bag_slots
        self.proj = nn.Linear(total_features, hidden_size)
        self.n_time_features = n_time_features
    
    def forward(self, fisheye: Tensor, time_features: Optional[Tensor] = None, bag_one_hot: Optional[Tensor] = None) -> Tensor:
        parts = [fisheye]
        if time_features is not None:
            parts.append(time_features)
        else:
            parts.append(torch.zeros(fisheye.shape[0], self.n_time_features, device=fisheye.device, dtype=fisheye.dtype))
        if bag_one_hot is not None:
            if bag_one_hot.shape[-1] < self.max_bag_slots:
                pad = torch.zeros(fisheye.shape[0], self.max_bag_slots - bag_one_hot.shape[-1], device=fisheye.device, dtype=fisheye.dtype)
                bag_one_hot = torch.cat([bag_one_hot, pad], dim=-1)
            elif bag_one_hot.shape[-1] > self.max_bag_slots:
                bag_one_hot = bag_one_hot[:, :self.max_bag_slots]
            parts.append(bag_one_hot)
        else:
            parts.append(torch.zeros(fisheye.shape[0], self.max_bag_slots, device=fisheye.device, dtype=fisheye.dtype))
        pancake = torch.cat(parts, dim=-1)
        return self.proj(pancake).unsqueeze(1)


def time_cyclical_features(bar_idx: int, bars_per_day: int = 288) -> List[float]:
    """DateShed-style cyclical time encoding. Returns sin/cos features for
    minute-of-hour, hour-of-day, day-of-week assuming 5-minute bars."""
    # bars_per_day = 288 for 5-min bars
    hour_len = 12  # bars per hour (5-min bars)
    features = []
    # Within-day phase
    day_phase = (bar_idx % bars_per_day) / bars_per_day
    features.append(math.sin(2 * math.pi * day_phase))
    features.append(math.cos(2 * math.pi * day_phase))
    # Within-hour phase
    hour_phase = (bar_idx % hour_len) / hour_len
    features.append(math.sin(2 * math.pi * hour_phase))
    features.append(math.cos(2 * math.pi * hour_phase))
    # Week phase (7 days)
    week_phase = (bar_idx % (bars_per_day * 7)) / (bars_per_day * 7)
    features.append(math.sin(2 * math.pi * week_phase))
    features.append(math.cos(2 * math.pi * week_phase))
    # Additional harmonics for intraday patterns
    features.append(math.sin(4 * math.pi * day_phase))
    features.append(math.cos(4 * math.pi * day_phase))
    features.append(math.sin(6 * math.pi * day_phase))
    features.append(math.cos(6 * math.pi * day_phase))
    # Raw bar-in-day and bar-in-hour normalized
    features.append(day_phase)
    features.append(hour_phase)
    # High-order harmonics for weekly structure
    features.append(math.sin(4 * math.pi * week_phase))
    features.append(math.cos(4 * math.pi * week_phase))
    features.append(math.sin(8 * math.pi * day_phase))
    features.append(math.cos(8 * math.pi * day_phase))
    features.append(math.sin(2 * math.pi * day_phase * 3))
    features.append(math.cos(2 * math.pi * day_phase * 3))
    return features


def stochastic_fisheye_boundaries(y_depth: int, x_pixels: int, curvature: float,
                                   rng: Optional[np.random.RandomState] = None) -> List[int]:
    """Fisheye boundaries with stochastic curvature perturbation.
    During training, curvature is jittered to teach the model math rigor
    instead of memorizing fixed bucket boundaries."""
    if rng is not None:
        # Jitter curvature by ±30% during training
        curvature = curvature * rng.uniform(0.7, 1.3)
        # Jitter y_depth by ±10% (but floor at x_pixels)
        y_depth = max(x_pixels, int(y_depth * rng.uniform(0.9, 1.1)))
    return fisheye_boundaries(y_depth, x_pixels, curvature)

class HRMBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, expansion: float = 4.0):
        super().__init__()
        self.self_attn = Attention(hidden_size, hidden_size // num_heads, num_heads, num_heads, causal=False)
        self.mlp = SwiGLU(hidden_size, expansion)
        self.norm_eps = 1e-5
    def forward(self, cos_sin, hidden_states: Tensor) -> Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin, hidden_states), self.norm_eps)
        return rms_norm(hidden_states + self.mlp(hidden_states), self.norm_eps)

class HRMReasoningModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, n_layers: int, expansion: float = 4.0):
        super().__init__()
        self.layers = nn.ModuleList([HRMBlock(hidden_size, num_heads, expansion) for _ in range(n_layers)])
    def forward(self, hidden_states: Tensor, input_injection: Tensor, cos_sin) -> Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin, hidden_states)
        return hidden_states

@dataclass
class HRMEdgeCarry:
    z_H: Tensor
    z_L: Tensor

class HRMEdgePredictor(nn.Module):
    def __init__(self, x_pixels=20, hidden_size=64, num_nodes=1, num_heads=4, H_layers=2, L_layers=2, H_cycles=2, L_cycles=2, expansion=4.0, rope_theta=10000.0, max_seq_len=64, max_bag_slots=32):
        super().__init__()
        self.x_pixels, self.hidden_size, self.num_nodes, self.num_heads = x_pixels, hidden_size, max(1, int(num_nodes)), num_heads
        self.H_layers, self.L_layers, self.H_cycles, self.L_cycles, self.expansion = H_layers, L_layers, H_cycles, L_cycles, expansion
        self.max_bag_slots = max_bag_slots
        
        self.embed = PancakeEmbedding(x_pixels, hidden_size, n_time_features=18, max_bag_slots=max_bag_slots)
        self.base_node_embed = nn.Embedding(self.num_nodes, hidden_size)
        self.quote_node_embed = nn.Embedding(self.num_nodes, hidden_size)
        self.pair_index_proj = nn.Linear(4, hidden_size)

        self.rotary_emb = RotaryEmbedding(hidden_size // num_heads, max_seq_len, rope_theta)
        self.H_level = HRMReasoningModule(hidden_size, num_heads, H_layers, expansion)
        self.L_level = HRMReasoningModule(hidden_size, num_heads, L_layers, expansion)

        self.register_buffer("H_init", trunc_normal_init_(torch.empty(hidden_size, dtype=torch.float32), 1), persistent=True)
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(hidden_size, dtype=torch.float32), 1), persistent=True)

        self.fraction_head = nn.Linear(hidden_size, 1)
        self.ptt_head = nn.Linear(hidden_size, 1)
        self.bid_head = nn.Linear(hidden_size, 1)
        self.sl_head = nn.Linear(hidden_size, 1)
        # Xavier init for heads so random weights produce nonzero signals from bar 1.
        # Zero-init backbone is fine; heads must break symmetry immediately.
        for head in [self.fraction_head, self.ptt_head, self.bid_head, self.sl_head]:
            nn.init.xavier_normal_(head.weight, gain=2.0)
            nn.init.uniform_(head.bias, -0.5, 0.5)

    def resize_node_embeddings(self, new_num_nodes: int):
        new_num_nodes = max(1, int(new_num_nodes))
        if new_num_nodes == self.num_nodes: return
        base, quote = nn.Embedding(new_num_nodes, self.hidden_size).to(self.base_node_embed.weight.device), nn.Embedding(new_num_nodes, self.hidden_size).to(self.quote_node_embed.weight.device)
        nn.init.zeros_(base.weight)
        nn.init.zeros_(quote.weight)
        rows_to_copy = min(self.num_nodes, new_num_nodes)
        with torch.no_grad():
            base.weight[:rows_to_copy] = self.base_node_embed.weight[:rows_to_copy]
            quote.weight[:rows_to_copy] = self.quote_node_embed.weight[:rows_to_copy]
        self.base_node_embed, self.quote_node_embed, self.num_nodes = base, quote, new_num_nodes

    def _pair_input_embedding(self, base_idx: Tensor, quote_idx: Tensor) -> Tensor:
        base_emb, quote_emb = self.base_node_embed(base_idx.long()), self.quote_node_embed(quote_idx.long())
        denom = float(max(1, self.num_nodes - 1))
        base_norm, quote_norm = base_idx.to(base_emb.dtype) / denom, quote_idx.to(base_emb.dtype) / denom
        pair_scalars = torch.stack([base_norm, quote_norm, base_norm - quote_norm, base_norm * quote_norm], dim=-1)
        return (base_emb + quote_emb + self.pair_index_proj(pair_scalars)).unsqueeze(1)

    def init_carry(self, batch_size: int, device: torch.device) -> HRMEdgeCarry:
        return HRMEdgeCarry(
            z_H=self.H_init.view(1, 1, self.hidden_size).expand(batch_size, -1, -1).clone().to(device),
            z_L=self.L_init.view(1, 1, self.hidden_size).expand(batch_size, -1, -1).clone().to(device),
        )

    def forward(self, fisheye: Tensor, base_idx: Optional[Tensor]=None, quote_idx: Optional[Tensor]=None, carry: Optional[HRMEdgeCarry]=None, time_features: Optional[Tensor]=None, bag_one_hot: Optional[Tensor]=None):
        input_emb = self.embed(fisheye, time_features, bag_one_hot)
        if base_idx is not None and quote_idx is not None:
            input_emb = input_emb + self._pair_input_embedding(base_idx, quote_idx)
        carry = carry or self.init_carry(fisheye.shape[0], fisheye.device)
        cos_sin = self.rotary_emb()
        z_H, z_L = carry.z_H, carry.z_L

        # HRM 1-step grad: run all cycles except the last in no_grad,
        # only the final L_step and H_step carry gradients.
        # This is the core HRM training insight from the AGI2 paper.
        with torch.no_grad():
            for h_step in range(self.H_cycles):
                for l_step in range(self.L_cycles):
                    if not ((h_step == self.H_cycles - 1) and (l_step == self.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin)
                if not (h_step == self.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, cos_sin)

        # Single differentiable step
        z_L = self.L_level(z_L, z_H + input_emb, cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin)

        z = z_H.squeeze(1)
        return (
            torch.tanh(self.fraction_head(z)).squeeze(-1),
            torch.sigmoid(self.ptt_head(z)).squeeze(-1),
            torch.tanh(self.bid_head(z)).squeeze(-1),
            torch.sigmoid(self.sl_head(z)).squeeze(-1),
            HRMEdgeCarry(z_H=z_H.detach(), z_L=z_L.detach()),
        )

    @staticmethod
    def _rotate_same_shape(param: Tensor, rotation: int) -> Tensor:
        if rotation == 0: return param.detach().clone()
        if param.dim() == 1: return torch.flip(param, dims=(0,))
        if param.dim() < 2: return param.detach().clone()
        if param.shape[-2] == param.shape[-1]: return torch.rot90(param, k={90: 1, 180: 2, 270: 3}[rotation], dims=(-2, -1))
        if rotation == 180: return torch.flip(param, dims=(-2, -1))
        if rotation == 90: return torch.flip(param, dims=(-2,))
        if rotation == 270: return torch.flip(param, dims=(-1,))
        raise ValueError(f"Unsupported rotation: {rotation}")

    @classmethod
    def _growth_variants_for_shape(cls, param: Tensor) -> List[Tensor]:
        """Return growth variants for a tensor shape.

        1D tensors use an axial 0/180 split.
        Square tensors use the full 4-way rotational set (0/180/90/270).
        Rectangular 2D tensors use a frame-style 0/270/90/180 set.
        """
        if param.dim() == 1 or param.dim() < 2:
            return [cls._rotate_same_shape(param, r) for r in (0, 180, 0, 180)]
        if param.shape[-2] == param.shape[-1]:
            return [cls._rotate_same_shape(param, r) for r in (0, 180, 90, 270)]
        return [cls._rotate_same_shape(param, r) for r in (0, 270, 90, 180)]

    @classmethod
    def _expand_axis_to(cls, param: Tensor, axis: int, target_size: int) -> Tensor:
        if param.shape[axis] == target_size: return param.detach().clone()
        if param.shape[axis] > target_size:
            slicer = [slice(None)] * param.dim(); slicer[axis] = slice(0, target_size)
            return param[tuple(slicer)].detach().clone()
        variants = cls._growth_variants_for_shape(param)
        pieces, remaining, v_idx =[], target_size, 0
        while remaining > 0:
            block = variants[v_idx % len(variants)]
            if block.shape[axis] <= remaining: pieces.append(block.detach().clone()); remaining -= block.shape[axis]
            else:
                slicer = [slice(None)] * block.dim(); slicer[axis] = slice(0, remaining)
                pieces.append(block[tuple(slicer)].detach().clone()); remaining = 0
            v_idx += 1
        return torch.cat(pieces, dim=axis)

    @classmethod
    def _expand_param_to_shape(cls, param: Tensor, target_shape: torch.Size) -> Tensor:
        if tuple(param.shape) == tuple(target_shape): return param.detach().clone()
        expanded = param.detach().clone()
        if expanded.dim() == 1: return cls._expand_axis_to(expanded, 0, target_shape[0])
        if expanded.shape[1] != target_shape[1]: expanded = cls._expand_axis_to(expanded, 1, target_shape[1])
        if expanded.shape[0] != target_shape[0]: expanded = cls._expand_axis_to(expanded, 0, target_shape[0])
        if tuple(expanded.shape) != tuple(target_shape): expanded = expanded[tuple(slice(0, s) for s in target_shape)].detach().clone()
        return expanded

    def grow_hidden_size(self, new_hidden_size: int):
        if new_hidden_size // self.hidden_size != 4: raise ValueError("Can only grow by 4×")
        grown = HRMEdgePredictor(self.x_pixels, new_hidden_size, self.num_nodes, max(1, new_hidden_size // 64), self.H_layers, self.L_layers, self.H_cycles, self.L_cycles, self.expansion, max_bag_slots=self.max_bag_slots).to(next(self.parameters()).device)
        expanded_sd = {k: self._expand_param_to_shape(self.state_dict()[k], v.shape).to(dtype=v.dtype, device=v.device) if k in self.state_dict() else v for k, v in grown.state_dict().items()}
        grown.load_state_dict(expanded_sd, strict=False)
        return grown

    def grow_layers(self, level: str = 'H'):
        module = getattr(self, f'{level}_level')
        new_layers =[]
        for rotation in (0, 180, 90, 270):
            for layer in module.layers:
                new_layer = HRMBlock(self.hidden_size, self.num_heads, self.expansion)
                new_layer.load_state_dict({k: self._rotate_same_shape(v, rotation) for k, v in layer.state_dict().items()})
                new_layers.append(new_layer)
        module.layers = nn.ModuleList(new_layers)
        if level == 'H': self.H_layers *= 4
        else: self.L_layers *= 4

    def grow_cycles(self, level: str = 'H'):
        if level == 'H': self.H_cycles *= 4
        else: self.L_cycles *= 4

def fisheye_boundaries(y_depth: int, x_pixels: int, curvature: float) -> List[int]:
    """Build candle-close compression buckets over the trailing `y_depth` closes.

    The fisheye is a close-history compressor: it emphasizes the most recent
    candle closes while still keeping older structure in coarser buckets.
    """
    if x_pixels <= 1: return [y_depth]
    boundaries, prev =[], 0
    for i in range(x_pixels):
        boundary = max(int(y_depth * ((i / (x_pixels - 1)) ** curvature)), prev + 1)
        boundaries.append(boundary)
        prev = boundary
    return boundaries

def fisheye_sample(candles, boundaries: List[int]) -> List[float]:
    if len(candles) == 0: return[0.0] * len(boundaries)
    results, prev_idx =[], 0
    for boundary in boundaries:
        bucket = candles[prev_idx:min(boundary, len(candles))]
        mean = np.mean(bucket) if len(bucket) > 0 else 0.0
        results.append((candles[-1] - mean) / mean if mean != 0 else 0.0)
        prev_idx = boundary
    return results

def _checkpoint_cas(metadata: Dict[str, object], state_dict: Dict[str, Tensor]) -> str:
    hasher = hashlib.sha256()
    hasher.update(CHECKPOINT_CAS_VERSION.encode("utf-8") + b"\0" + json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8") + b"\0")
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name].detach().cpu().contiguous()
        hasher.update(name.encode("utf-8") + b"\0" + str(tuple(tensor.shape)).encode("utf-8") + b"\0" + str(tensor.dtype).encode("utf-8") + b"\0" + tensor.numpy().tobytes() + b"\0")
    return hasher.hexdigest()

class HierarchicalReasoningModel:
    def __init__(self, n_edges=0, learning_rate=0.005, y_depth=200, x_pixels=20, curvature=2.0, h_dim=4, z_dim=4, prediction_depth=1, H_layers=2, L_layers=2, H_cycles=2, L_cycles=2, device="auto", max_bag_slots=32, stochastic_fisheye=True, **kwargs):
        self.y_depth, self.x_pixels, self._lr, self.curvature = y_depth, x_pixels, learning_rate, curvature
        self.h_dim, self.z_dim, self.prediction_depth = h_dim, z_dim, prediction_depth
        self.H_layers, self.L_layers, self.H_cycles, self.L_cycles = H_layers, L_layers, H_cycles, L_cycles
        self.max_bag_slots, self.stochastic_fisheye = max_bag_slots, stochastic_fisheye
        self.device_preference = device
        self.edge_names, self.node_names, self.node_to_idx, self.num_nodes = [],[], {}, 0
        self._model, self._optimizer, self._device = None, None, get_device(self.device_preference)
        self._graph = None  # pandas-only: model reads directly from graph.edges DataFrames
        self._edge_bar_idx = {}  # last bar_idx seen per edge (for bar-counting)
        self._prediction_queue, self._carry, self._observed_edges_for_bar, self._kline_view, self._kline_live = {}, {}, [], {}, {}
        self._max_history = max(y_depth + 100, 500)
        self._last_price_bar_idx = None
        self._fisheye_boundaries = fisheye_boundaries(self.y_depth, self.x_pixels, self.curvature)
        self._rng = np.random.RandomState(42)
        self._profile_enabled, self._profile_stats = False, {}
        self.reset_profile_stats()

    def _checkpoint_metadata(self) -> Dict[str, object]:
        return {'model_class': 'HRMEdgePredictor', 'x_pixels': self.x_pixels, 'y_depth': self.y_depth, 'curvature': self.curvature, 'h_dim': self.h_dim, 'z_dim': self.z_dim, 'num_nodes': int(self.num_nodes or 0), 'prediction_depth': self.prediction_depth, 'H_layers': self.H_layers, 'L_layers': self.L_layers, 'H_cycles': self.H_cycles, 'L_cycles': self.L_cycles, 'max_bag_slots': self.max_bag_slots, 'stochastic_fisheye': self.stochastic_fisheye}

    def model_cas_signature(self) -> Optional[str]: return _checkpoint_cas(self._checkpoint_metadata(), self._model.state_dict()) if self._model else None

    def _reset_edge_runtime_state(self):
        self._graph = None
        self._edge_bar_idx = {e: -1 for e in self.edge_names}
        self._prediction_queue, self._carry, self._observed_edges_for_bar, self._last_price_bar_idx, self._kline_view, self._kline_live = {e:[] for e in self.edge_names}, {e: None for e in self.edge_names},[], None, {e: {} for e in self.edge_names}, {e: False for e in self.edge_names}

    def _build_optimizer(self):
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr, weight_decay=0.01)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self._optimizer, T_0=50, T_mult=2, eta_min=self._lr * 0.01)

    def set_lr_multiplier(self, multiplier: float):
        """Scale the base LR in optimizer param groups. Resets on multiplier=1.0."""
        if self._optimizer is None: return
        target_lr = self._lr * float(multiplier)
        for pg in self._optimizer.param_groups:
            pg['lr'] = target_lr

    def _build_model(self):
        self._device = get_device(self.device_preference)
        self._model = HRMEdgePredictor(self.x_pixels, self.h_dim, max(1, self.num_nodes or len(self.node_names) or 1), max(1, self.h_dim // 64), self.H_layers, self.L_layers, self.H_cycles, self.L_cycles, max_bag_slots=self.max_bag_slots).to(self._device)
        self._build_optimizer()
        self._max_history = max(self.y_depth + 100, 500)
        self._reset_edge_runtime_state()

    def _apply_checkpoint_config(self, state: Dict):
        for k, v in state.items():
            if k in['x_pixels', 'y_depth', 'curvature', 'h_dim', 'z_dim', 'num_nodes', 'prediction_depth', 'H_layers', 'L_layers', 'H_cycles', 'L_cycles', 'max_bag_slots', 'stochastic_fisheye'] and v is not None: setattr(self, k, v)
        self._max_history = max(self.y_depth + 100, 500)
        self._fisheye_boundaries = fisheye_boundaries(self.y_depth, self.x_pixels, self.curvature)

    def register_edges(self, edges: List[Tuple[str, ...]]):
        self.edge_names = edges
        node_set = set()
        for e in edges: node_set.update([e[0], e[1]] if len(e)==2 else [e[1], e[2]])
        if "USD" in node_set: node_set.remove("USD"); self.node_names = ["USD"] + sorted(node_set)
        else: self.node_names = sorted(node_set)
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}
        self.num_nodes = len(self.node_names)
        if self._model is None: self._build_model()
        else: self._model.resize_node_embeddings(max(1, self.num_nodes)); self._reset_edge_runtime_state()

    def _edge_index_batch(self, edges: List[Tuple[str, ...]]) -> Tuple[Tensor, Tensor]:
        base_idx, quote_idx = zip(*[(self.node_to_idx[str(e[0] if len(e)==2 else e[1])], self.node_to_idx[str(e[1] if len(e)==2 else e[2])]) for e in edges])
        return torch.as_tensor(np.asarray(base_idx, dtype=np.int64), device=self._device), torch.as_tensor(np.asarray(quote_idx, dtype=np.int64), device=self._device)

    def build_bag_model_frames(self, graph, bar_idx: int, *, value_asset: str = "USD", free_qty=None, reserved_qty=None, route_discount=None):
        return build_bag_model_frames(graph, bar_idx=bar_idx, edge_names=self.edge_names, node_to_idx=self.node_to_idx, value_asset=value_asset, edge_fisheyes={e: self._get_fisheye(e) for e in self.edge_names}, edge_carries={e: self._carry.get(e) for e in self.edge_names}, free_qty=free_qty, reserved_qty=reserved_qty, route_discount=route_discount)

    @staticmethod
    def bind_spend_budgets(pred_df: pd.DataFrame, node_state_df: pd.DataFrame) -> pd.DataFrame:
        return bind_spend_budgets(pred_df, node_state_df)

    def _obs_count(self, graph, edge: Tuple[str, ...], bar_idx: int) -> int:
        """Count of DataFrame rows seen up to and including bar_idx. Pandas-only."""
        df = graph.edges.get(edge)
        if df is None or bar_idx < 0 or bar_idx >= len(graph.common_timestamps):
            return 0
        ts = graph.common_timestamps[bar_idx]
        if ts not in df.index:
            return 0
        return df.index.get_loc(ts) + 1

    def _get_closes(self, graph, edge: Tuple[str, ...], bar_idx: int) -> np.ndarray:
        """Read close prices directly from graph.edges DataFrame. Pandas-only."""
        df = graph.edges.get(edge)
        if df is None or bar_idx < 0:
            return np.array([], dtype=np.float32)
        ts = graph.common_timestamps[bar_idx] if bar_idx < len(graph.common_timestamps) else None
        if ts is None or ts not in df.index:
            return np.array([], dtype=np.float32)
        # Slice up to and including bar_idx
        loc = df.index.get_loc(ts)
        if loc is None:
            return np.array([], dtype=np.float32)
        start = max(0, loc + 1 - self._max_history)
        rows = df.iloc[start:loc + 1]
        if hasattr(graph, 'edge_price_components'):
            closes = []
            for _, row in rows.iterrows():
                comp = graph.edge_price_components(edge, row)
                closes.append(float(comp["close"]))
            return np.asarray(closes, dtype=np.float32)
        c = rows['close'].values.astype(np.float32)
        if getattr(graph, 'edge_is_inverted', {}).get(edge, False):
            c = np.where(c > 0, 1.0 / c, c)
        return c

    def _get_fisheye(self, graph, edge: Tuple[str, ...], bar_idx: int) -> List[float]:
        closes = self._get_closes(graph, edge, bar_idx)
        if len(closes) < self.y_depth: return [0.0] * self.x_pixels
        rng = self._rng if self.stochastic_fisheye else None
        boundaries = stochastic_fisheye_boundaries(self.y_depth, self.x_pixels, self.curvature, rng)
        values = fisheye_sample(closes[-self.y_depth:], boundaries)[:self.x_pixels]
        return values + [0.0] * (self.x_pixels - len(values))

    def _bag_one_hot(self, active_edges: List[Tuple[str, ...]]) -> Tensor:
        """Build bag slot one-hot: which edges in the bag are active this bar.
        Random ordering (shuffle) to prevent position bias."""
        edge_set = set(active_edges)
        n_slots = min(len(self.edge_names), self.max_bag_slots)
        indices = list(range(n_slots))
        self._rng.shuffle(indices)
        one_hot = torch.zeros(len(active_edges), self.max_bag_slots, device=self._device)
        for i, edge in enumerate(active_edges):
            if edge in self.edge_names:
                slot = self.edge_names.index(edge)
                if slot < self.max_bag_slots:
                    one_hot[i, indices[slot] % self.max_bag_slots] = 1.0
        return one_hot

    def _time_features_batch(self, bar_idx: int, batch_size: int) -> Tensor:
        """Compute time cyclical features for a batch."""
        features = time_cyclical_features(bar_idx)
        return torch.tensor([features] * batch_size, dtype=torch.float32, device=self._device)

    def reset_profile_stats(self): self._profile_stats = {k: 0.0 for k in['bars_observed', 'stale_frames_dropped', 'update_prices_seconds', 'predict_batches', 'predict_edges', 'predict_prepare_seconds', 'predict_forward_seconds', 'update_batches', 'update_edges', 'update_prepare_seconds', 'update_forward_backward_seconds']}
    def set_profile_enabled(self, enabled: bool): self._profile_enabled = enabled; self.reset_profile_stats()
    def get_profile_stats(self) -> Dict[str, float]: return {**self._profile_stats, 'device_type': self._device.type}
    def _record_profile(self, key: str, value: float):
        if self._profile_enabled: self._profile_stats[key] += value
    def _sync_device_for_profile(self):
        if not self._profile_enabled: return
        if self._device.type == "mps": torch.mps.synchronize()
        elif self._device.type == "cuda": torch.cuda.synchronize(self._device)

    def ready_for_prediction(self, bar_idx: Optional[int] = None) -> bool:
        if bar_idx is not None and self._last_price_bar_idx != bar_idx: return False
        if self._graph is None: return False
        bi = self._last_price_bar_idx or 0
        return any(len(self._get_closes(self._graph, e, bi)) >= self.y_depth for e in self._observed_edges_for_bar)

    def _matured_prediction_frame(self, edge: Tuple[str, str], graph=None) -> Optional[Dict]:
        queue = self._prediction_queue.get(edge,[])
        bar_idx = self._edge_bar_idx.get(edge, -1)
        observed = self._obs_count(graph, edge, bar_idx) if graph is not None else 0
        while queue and observed - queue[0]['observation_count'] > self.prediction_depth:
            queue.pop(0)
            self._record_profile('stale_frames_dropped', 1.0)
        if not queue: return None
        frame = queue[0]
        if observed - frame['observation_count'] >= self.prediction_depth: return frame
        if graph is not None and frame.get('start_price', 0) > 0:
            closes = self._get_closes(graph, edge, bar_idx)
            if len(closes) > 0:
                cum_return = (closes[-1] / frame['start_price']) - 1.0
                st = getattr(graph, "edge_state", {}).get(edge)
                if st and (cum_return >= getattr(st, "ptt", float('inf')) or cum_return <= getattr(st, "stop", float('-inf'))): return frame
        return None

    def ready_for_update(self, bar_idx: Optional[int] = None, actual_accels=None, graph=None) -> bool:
        if bar_idx is not None and self._last_price_bar_idx != bar_idx: return False
        edges = actual_accels.keys() if actual_accels else self._observed_edges_for_bar
        return any(self._matured_prediction_frame(e, graph) is not None for e in edges)

    def force_update(self, graph, edge_accels, bar_idx: int, actual_velocities=None, hit_ptt=None, hit_stop=None) -> Optional[float]:
        """Force a training step from current observations, bypassing the prediction queue.

        Used when the model has been idle (no matured predictions) and needs to restart
        gradient flow. Synthesizes a target from the current bar's velocity.
        """
        if not self.edge_names or self._model is None or self._optimizer is None:
            return None
        if bar_idx >= 0 and self._last_price_bar_idx != bar_idx:
            return None
        # Find edges with enough data to build fisheye (pandas check)
        usable = [e for e in edge_accels if e in self._edge_bar_idx and len(self._get_closes(graph, e, bar_idx)) >= self.y_depth]
        if not usable:
            return None
        fisheye_rows = [self._get_fisheye(graph, e, bar_idx) for e in usable]
        base_idx_batch, quote_idx_batch = self._edge_index_batch(usable)
        time_feat = self._time_features_batch(bar_idx, len(usable))
        bag_oh = self._bag_one_hot(usable)
        input_carry = self._model.init_carry(len(usable), self._device)
        for idx, e in enumerate(usable):
            if self._carry.get(e):
                input_carry.z_H[idx:idx+1], input_carry.z_L[idx:idx+1] = self._carry[e].z_H, self._carry[e].z_L
        fisheye_batch = torch.as_tensor(np.asarray(fisheye_rows, dtype=np.float32), device=self._device)
        # Target: current velocity from edge_accels
        vel_targets = [edge_accels.get(e, 0.0) for e in usable]
        velocities_tensor = torch.as_tensor(np.asarray(vel_targets, dtype=np.float32), device=self._device)
        self._optimizer.zero_grad(set_to_none=True)
        pred_fraction, pred_tpp, pred_bid, pred_sl, output_carry = self._model(
            fisheye_batch, base_idx_batch, quote_idx_batch, input_carry, time_feat, bag_oh)
        for idx, e in enumerate(usable):
            self._carry[e] = HRMEdgeCarry(
                z_H=output_carry.z_H[idx:idx+1].detach().clone(),
                z_L=output_carry.z_L[idx:idx+1].detach().clone())
            # Also enqueue so normal pipeline can resume
            closes = self._get_closes(graph, e, bar_idx)
            obs_count = self._obs_count(graph, e, bar_idx)
            self._prediction_queue[e].append({
                'fisheye': fisheye_rows[idx], 'bar_idx': bar_idx,
                'observation_count': obs_count,
                'carry': HRMEdgeCarry(
                    z_H=input_carry.z_H[idx:idx+1].detach().clone(),
                    z_L=input_carry.z_L[idx:idx+1].detach().clone()),
                'start_price': float(closes[-1]) if len(closes) > 0 else 1.0,
            })
        import torch.nn.functional as F
        allocations = pred_bid * torch.abs(pred_fraction)
        hit_ptt_tensor = torch.as_tensor(
            np.asarray([1.0 if (hit_ptt or {}).get(e, False) else 0.0 for e in usable], dtype=np.float32),
            device=self._device)
        hit_stop_tensor = torch.as_tensor(
            np.asarray([1.0 if (hit_stop or {}).get(e, False) else 0.0 for e in usable], dtype=np.float32),
            device=self._device)
        reward = allocations * velocities_tensor * 1000.0
        pnl_loss = -torch.mean(torch.log1p(F.relu(reward)) - 2.0 * torch.log1p(F.relu(-reward)))
        tpp_loss = F.binary_cross_entropy(pred_tpp, hit_ptt_tensor)
        sl_loss = F.binary_cross_entropy(pred_sl, hit_stop_tensor)
        loss = (pnl_loss * 10.0) + tpp_loss + sl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()
        self._scheduler.step()
        self._record_profile('update_batches', 1.0)
        self._record_profile('update_edges', float(len(usable)))
        return float(loss.item())

    def update_prices(self, graph, bar_idx: int) -> List[Tuple[str, str]]:
        start = time.perf_counter() if self._profile_enabled else None
        if bar_idx < 0 or bar_idx >= len(graph.common_timestamps): self._last_price_bar_idx = None; self._observed_edges_for_bar = []; return []
        # Pandas-only: store graph reference for all reads
        self._graph = graph
        ts = graph.common_timestamps[bar_idx]
        observed_edges = []
        for edge in self.edge_names:
            df = graph.edges.get(edge)
            if df is None or ts not in df.index: continue
            self._edge_bar_idx[edge] = bar_idx
            observed_edges.append(edge)
        # Store kline_view pancake for each observed edge (stateless per bar)
        if observed_edges:
            try:
                from kline_view import decorate_view, is_live_bar
                for edge in observed_edges:
                    self._kline_view[edge] = decorate_view(graph, edge, bar_idx, horizon_width=self.x_pixels)
                    self._kline_live[edge] = is_live_bar(graph, edge, bar_idx)
            except ImportError:
                pass
        self._last_price_bar_idx, self._observed_edges_for_bar = bar_idx, observed_edges
        self._record_profile('bars_observed', 1.0)
        if start is not None: self._record_profile('update_prices_seconds', time.perf_counter() - start)
        return observed_edges

    def predict(
        self, graph, bar_idx: int = -1
    ) -> Dict[Tuple[str, str], Tuple[float, float, float, float]]:
        if not self.edge_names or self._model is None: return {}
        ready_edges = [e for e in self._observed_edges_for_bar if len(self._get_closes(graph, e, bar_idx)) >= self.y_depth]
        if not ready_edges: return {}
        prep_start = time.perf_counter() if self._profile_enabled else None
        fisheye_rows = [self._get_fisheye(graph, e, bar_idx) for e in ready_edges]
        fisheye_batch = torch.as_tensor(np.asarray(fisheye_rows, dtype=np.float32), device=self._device)
        base_idx_batch, quote_idx_batch = self._edge_index_batch(ready_edges)
        time_feat = self._time_features_batch(bar_idx, len(ready_edges))
        bag_oh = self._bag_one_hot(ready_edges)
        input_carry = self._model.init_carry(len(ready_edges), self._device)
        for idx, e in enumerate(ready_edges):
            if self._carry.get(e): input_carry.z_H[idx:idx+1], input_carry.z_L[idx:idx+1] = self._carry[e].z_H, self._carry[e].z_L
        if prep_start: self._record_profile('predict_prepare_seconds', time.perf_counter() - prep_start)
        self._sync_device_for_profile()
        fwd_start = time.perf_counter() if self._profile_enabled else None
        with torch.no_grad():
            fraction, tpp_out, bid_out, sl_out, output_carry = self._model(fisheye_batch, base_idx_batch, quote_idx_batch, input_carry, time_feat, bag_oh)
        self._sync_device_for_profile()
        if fwd_start: self._record_profile('predict_forward_seconds', time.perf_counter() - fwd_start)
        self._record_profile('predict_batches', 1.0); self._record_profile('predict_edges', float(len(ready_edges)))
        fractions = fraction.detach().cpu().tolist()
        tpps = tpp_out.detach().cpu().tolist()
        bids = bid_out.detach().cpu().tolist()
        sls = sl_out.detach().cpu().tolist()
        predictions = {}
        for idx, e in enumerate(ready_edges):
            predictions[e] = (
                float(fractions[idx]),
                float(tpps[idx]),
                float(bids[idx]),
                float(sls[idx]),
            )
            self._carry[e] = HRMEdgeCarry(z_H=output_carry.z_H[idx:idx+1].detach().clone(), z_L=output_carry.z_L[idx:idx+1].detach().clone())
            closes_e = self._get_closes(graph, e, bar_idx)
            obs_count = self._obs_count(graph, e, bar_idx)
            self._prediction_queue[e].append({'fisheye': fisheye_rows[idx], 'bar_idx': bar_idx, 'observation_count': obs_count, 'carry': HRMEdgeCarry(z_H=input_carry.z_H[idx:idx+1].detach().clone(), z_L=input_carry.z_L[idx:idx+1].detach().clone()), 'start_price': float(closes_e[-1]) if len(closes_e) > 0 else 1.0})
        return predictions

    def update(self, graph, actual_accels: Dict[Tuple[str, str], float], bar_idx: int = -1, actual_velocities=None, hit_ptt=None, hit_stop=None) -> Optional[float]:
        if not self.edge_names or self._model is None or self._optimizer is None or hit_ptt is None or hit_stop is None or (bar_idx >= 0 and self._last_price_bar_idx != bar_idx): return None
        prep_start = time.perf_counter() if self._profile_enabled else None
        matured_edges, fisheye_rows, carry_rows, velocity_targets = [], [], [],[]
        for edge in actual_accels:
            frame = self._matured_prediction_frame(edge, graph)
            if not frame: continue
            closes_e = self._get_closes(graph, edge, bar_idx)
            obs_count = self._obs_count(graph, edge, bar_idx)
            vel = ((float(closes_e[-1]) / frame.get('start_price', 1.0)) - 1.0) / max(1, obs_count - frame['observation_count']) if frame.get('start_price', 0) > 0 and len(closes_e) > 0 else 0.0
            matured_edges.append(edge); fisheye_rows.append(frame['fisheye']); carry_rows.append(frame['carry']); velocity_targets.append(vel)
        if not matured_edges: return None
        fisheye_batch, velocities_tensor = torch.as_tensor(np.asarray(fisheye_rows, dtype=np.float32), device=self._device), torch.as_tensor(np.asarray(velocity_targets, dtype=np.float32), device=self._device)
        base_idx_batch, quote_idx_batch = self._edge_index_batch(matured_edges)
        time_feat = self._time_features_batch(bar_idx, len(matured_edges))
        bag_oh = self._bag_one_hot(matured_edges)
        carry_batch = HRMEdgeCarry(z_H=torch.cat([c.z_H for c in carry_rows], dim=0), z_L=torch.cat([c.z_L for c in carry_rows], dim=0))
        if prep_start: self._record_profile('update_prepare_seconds', time.perf_counter() - prep_start)
        self._optimizer.zero_grad(set_to_none=True)
        self._sync_device_for_profile()
        fwd_start = time.perf_counter() if self._profile_enabled else None
        # Fraction is position size (tanh); bid is direction (tanh, +buy/-sell); tpp/sl are event heads.
        pred_fraction, pred_tpp, pred_bid, pred_sl, _ = self._model(fisheye_batch, base_idx_batch, quote_idx_batch, carry_batch, time_feat, bag_oh)
        allocations = pred_bid * torch.abs(pred_fraction)

        hit_ptt_tensor = torch.as_tensor(
            np.asarray([1.0 if hit_ptt.get(edge, False) else 0.0 for edge in matured_edges], dtype=np.float32),
            device=self._device,
        )
        hit_stop_tensor = torch.as_tensor(
            np.asarray([1.0 if hit_stop.get(edge, False) else 0.0 for edge in matured_edges], dtype=np.float32),
            device=self._device,
        )

        # Reward-based loss: the model learns from realized PnL only.
        # allocation * velocity = profit (positive = money made).
        # We scale velocity by 1000 so the gradient signal is meaningful.
        reward = allocations * velocities_tensor * 1000.0

        # PnL loss: maximize reward. Use log1p for stability, asymmetric
        # so losses hurt more than wins feel good (risk-aware).
        pnl_positive = torch.log1p(F.relu(reward))
        pnl_negative = -2.0 * torch.log1p(F.relu(-reward))
        pnl_loss = -torch.mean(pnl_positive + pnl_negative)

        # TPP/SL event heads train against realized events (auxiliary).
        tpp_loss = F.binary_cross_entropy(pred_tpp, hit_ptt_tensor)
        sl_loss = F.binary_cross_entropy(pred_sl, hit_stop_tensor)

        loss = (pnl_loss * 10.0) + tpp_loss + sl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()
        self._scheduler.step()
        if fwd_start: self._record_profile('update_forward_backward_seconds', time.perf_counter() - fwd_start)
        self._record_profile('update_batches', 1.0); self._record_profile('update_edges', float(len(matured_edges)))
        for e in matured_edges: self._prediction_queue[e].pop(0)
        return float(loss.item())

    def save(self, path: str = "model_weights.pt", checkpoint_type: str = "pretrained"):
        if self._model: torch.save({'model': self._model.state_dict(), 'x_pixels': self.x_pixels, 'y_depth': self.y_depth, 'curvature': self.curvature, 'h_dim': self.h_dim, 'z_dim': self.z_dim, 'num_nodes': int(self.num_nodes or 0), 'prediction_depth': self.prediction_depth, 'H_layers': self.H_layers, 'L_layers': self.L_layers, 'H_cycles': self.H_cycles, 'L_cycles': self.L_cycles, 'max_bag_slots': self.max_bag_slots, 'stochastic_fisheye': self.stochastic_fisheye, 'checkpoint_timestamp': __import__('datetime').datetime.now().isoformat(), 'checkpoint_type': checkpoint_type, 'model_cas': self.model_cas_signature()}, path)

    def load(self, path: str = "model_weights.pt"):
        if not Path(path).exists(): return
        try: state = torch.load(path, weights_only=True, map_location=self._device)
        except: state = torch.load(path, weights_only=False, map_location=self._device)
        if not isinstance(state, dict) or 'model' not in state: return
        self._apply_checkpoint_config(state)
        self._build_model()
        self._model.load_state_dict(state['model'], strict=False)

    def grow(self, dim: str):
        if not self._model: return
        if dim == 'h':
            self._model = self._model.grow_hidden_size(SQUARE_CUBE_SIZES[SQUARE_CUBE_SIZES.index(self._model.hidden_size) + 1] if self._model.hidden_size in SQUARE_CUBE_SIZES and SQUARE_CUBE_SIZES.index(self._model.hidden_size) + 1 < len(SQUARE_CUBE_SIZES) else self._model.hidden_size * 4).to(self._device)
            self.h_dim = self._model.hidden_size
        elif dim in ('H', 'L'):
            self._model.grow_layers(dim)
            if dim == 'H': self.H_layers = self._model.H_layers
            else: self.L_layers = self._model.L_layers
        elif dim in ('Hc', 'Lc'):
            self._model.grow_cycles('H' if dim == 'Hc' else 'L')
            if dim == 'Hc': self.H_cycles = self._model.H_cycles
            else: self.L_cycles = self._model.L_cycles
        self._build_optimizer()
        self._reset_edge_runtime_state()
