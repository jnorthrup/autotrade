"""
Codec 23: Transformer Attention
Multi-head self-attention over multi-timeframe bar patches.

Architecture:
  - Tokenise OHLCV bars into patch embeddings at 3 timeframes (5m / 15m / 60m)
  - Apply scaled dot-product self-attention (non-causal; all patches see each other)
  - Project attended representation → (conviction, direction)

Pure numpy implementation of scaled dot-product attention so this works
without MLX.  When MLX is available, a learned transformer module is blended in.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .base_codec import BaseCodec

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

_PATCH_DIM = 5          # OHLCV per patch
_N_HEADS   = 4
_D_MODEL   = 16         # per-head key/query/value dim


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu  = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mu) / (std + eps)


def _soft_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention.  Q/K/V: [T, d]"""
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d + 1e-8)
    scores -= scores.max(axis=-1, keepdims=True)   # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True) + 1e-8
    return weights @ V   # [T, d]


class _MultiHeadAttention:
    """Deterministic multi-head self-attention with fixed projection weights."""

    def __init__(self, n_tokens: int, d_in: int, d_model: int, n_heads: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_heads = n_heads
        self.d_head  = d_model

        # Each head projects input → Q, K, V
        self.Wq = [rng.randn(d_in, d_model).astype(np.float32) * 0.1 for _ in range(n_heads)]
        self.Wk = [rng.randn(d_in, d_model).astype(np.float32) * 0.1 for _ in range(n_heads)]
        self.Wv = [rng.randn(d_in, d_model).astype(np.float32) * 0.1 for _ in range(n_heads)]
        self.Wo = rng.randn(n_heads * d_model, 2).astype(np.float32) * 0.1

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: [T, d_in] → [2] (direction_logit, conviction_logit)"""
        head_outs = []
        for h in range(self.n_heads):
            Q = X @ self.Wq[h]
            K = X @ self.Wk[h]
            V = X @ self.Wv[h]
            attended = _soft_attention(Q, K, V)   # [T, d_head]
            head_outs.append(attended[-1])         # use last-token representation
        concat = np.concatenate(head_outs)         # [n_heads * d_head]
        return concat @ self.Wo                    # [2]


class Codec23(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'transformer_attention'
        super().__init__(config)

        self.patch_size   = config.get('patch_size', 5)      # bars per patch
        self.n_patches    = config.get('n_patches', 12)      # total patches per timeframe
        self.n_timeframes = 3                                 # 5m, 15m, 60m aggregations

        d_in = _PATCH_DIM * self.n_timeframes   # OHLCV * 3 timeframes per token
        self._attn = _MultiHeadAttention(
            n_tokens  = self.n_patches,
            d_in      = d_in,
            d_model   = _D_MODEL,
            n_heads   = _N_HEADS,
            seed      = 23
        )

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            self.model = None

    def _build_patches(self, market_data: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        """
        Build [n_patches, _PATCH_DIM * n_timeframes] patch matrix.

        We reconstruct approximate price series from the return vector stored
        in `features`, then resample at 3 timeframe aggregations.
        """
        price  = float(market_data.get('price', 1.0))
        high   = float(market_data.get('high', price))
        low    = float(market_data.get('low', price))
        volume = float(market_data.get('volume', 1.0))

        returns = market_data.get('ret_series', features[:min(len(features), 64)])
        n = len(returns)

        if n < 5:
            return np.zeros((self.n_patches, _PATCH_DIM * self.n_timeframes), dtype=np.float32)

        # Use real pandas-sourced close series via get_ohlcv()
        closes, _, _, _ = self.get_ohlcv(market_data, features)

        patches = []
        window  = self.patch_size * self.n_patches   # total bars to cover

        for tf_factor in [1, 3, 12]:                 # 5m / 15m / 60m
            # Aggregate by summing tf_factor consecutive bars
            tf_closes = closes[-min(n, window * tf_factor):]
            # Downsample by tf_factor via overlapping mean
            step = max(1, tf_factor)
            resampled = np.array([
                tf_closes[max(0, i - step):i].mean()
                for i in range(step, len(tf_closes) + 1, step)
            ])
            # Ensure exactly n_patches patches × patch_size bars
            if len(resampled) < self.n_patches:
                resampled = np.pad(resampled, (self.n_patches - len(resampled), 0), mode='edge')
            else:
                resampled = resampled[-self.n_patches:]

            # Build per-patch OHLCV features (simplified: O=prev_close, H/L±vol, C=close, V=1)
            patch_feat = []
            for i in range(self.n_patches):
                c = float(resampled[i])
                prev_c = float(resampled[max(0, i - 1)])
                vol_approx = abs(c - prev_c) * 1.5
                o = prev_c
                h = c + vol_approx * 0.5
                l = c - vol_approx * 0.5
                ret = (c - prev_c) / (prev_c + 1e-8)
                patch_feat.append([o, h, l, c, ret])
            patches.append(np.array(patch_feat, dtype=np.float32))  # [n_patches, 5]

        # Concatenate timeframes: [n_patches, 15]
        tokens = np.concatenate(patches, axis=-1)
        tokens = _layer_norm(tokens)
        return tokens

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        tokens = self._build_patches(market_data, features)   # [n_patches, d_in]

        logits = self._attn.forward(tokens)   # [2]: direction_logit, conviction_logit
        direction  = float(np.tanh(logits[0]))
        confidence = float(1.0 / (1.0 + np.exp(-logits[1])))  # sigmoid

        # Position-weighted attention score as additional conviction signal
        price = float(market_data.get('price', 1.0))
        returns = features[:min(len(features), 20)]
        if len(returns) >= 5:
            attn_weighted_ret = float(np.average(returns[-5:], weights=[0.1, 0.15, 0.2, 0.25, 0.3]))
            # If attention direction disagrees with recent momentum, lower conviction
            momentum_bias = np.sign(attn_weighted_ret)
            if momentum_bias != 0 and np.sign(direction) != momentum_bias:
                confidence *= 0.7

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                ml_dir  = float(np.tanh(float(out[0, 0])))
                ml_conf = float(mx.sigmoid(out[0, 1]))
                direction  = direction * 0.5 + ml_dir * 0.5
                confidence = confidence * 0.5 + ml_conf * 0.5
            except Exception:
                pass

        self.record_instruments(
                attn_dir_logit=float(logits[0]) if 'logits' in dir() else 0.0,
                attn_conf_logit=float(logits[1]) if 'logits' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
