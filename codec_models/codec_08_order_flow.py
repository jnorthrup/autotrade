"""
Codec 08: Order Flow
Infers buyers vs sellers from tick-level bar construction.

Without a live LOB feed, order flow is estimated from the OHLCV bar structure:
  - Delta proxy = (close - open) / (high - low)  ∈ [-1, +1]
    Positive → buyers dominated the bar; negative → sellers
  - Cumulative delta (CVD proxy): rolling sum of bar deltas
  - Volume velocity: rate of change of implied volume
  - Absorption detection: large range but small net delta → absorption (fade signal)

Reference: Steidlmayer's Market Profile / Harris Order Flow.
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


class Codec08(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'order_flow'
        super().__init__(config)

        self.cvd_window    = config.get('cvd_window', 20)
        self.absorb_thresh = config.get('absorb_thresh', 0.15)

        # Stateful CVD accumulator across bars
        self._cvd_history: list = []

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            self.model = None

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price  = float(market_data.get('price', 1.0))
        high   = float(market_data.get('high', price))
        low    = float(market_data.get('low', price))
        volume = float(market_data.get('volume', 1.0))
        atr    = float(market_data.get('atr_14', (high - low) + 1e-8))

        bar_range = high - low + 1e-8

        # ── Bar delta proxy ────────────────────────────────────────────────
        # Reconstruct open from previous close using return
        returns = features[:min(len(features), 64)]
        if len(returns) >= 2:
            prev_ret = float(returns[-2])
            open_proxy = price / (1.0 + float(returns[-1]) + 1e-8)
        else:
            open_proxy = (high + low) / 2.0

        net_move   = price - open_proxy
        bar_delta  = net_move / bar_range           # ∈ [-1, +1]

        # Volume-weighted delta
        rel_vol    = min(volume / (atr * 1e4 + 1e-8), 3.0)   # rough normalisation
        vol_delta  = bar_delta * rel_vol

        # ── CVD (Cumulative Volume Delta) ──────────────────────────────────
        self._cvd_history.append(vol_delta)
        if len(self._cvd_history) > self.cvd_window:
            self._cvd_history = self._cvd_history[-self.cvd_window:]

        cvd  = float(np.sum(self._cvd_history))
        cvd_mean = float(np.mean(self._cvd_history))
        cvd_std  = float(np.std(self._cvd_history)) + 1e-8
        cvd_z    = (cvd - cvd_mean * len(self._cvd_history)) / (cvd_std * np.sqrt(len(self._cvd_history)) + 1e-8)

        # ── Absorption detection ───────────────────────────────────────────
        # Large bar range but small |net_move| → absorbing supply/demand
        absorption = (bar_range / (atr + 1e-8)) > 1.5 and abs(bar_delta) < self.absorb_thresh
        if absorption:
            # Fade: price was pushed to extreme but absorbed → likely reversal
            direction  = -float(np.sign(net_move)) if net_move != 0 else 0.0
            confidence = min(1.0, 0.6 + abs(bar_delta) * 0.2)
        else:
            # Follow the CVD trend
            direction  = float(np.sign(cvd_z)) if abs(cvd_z) > 0.5 else 0.0
            confidence = min(1.0, 0.2 + abs(cvd_z) * 0.2 + abs(bar_delta) * 0.3)

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                direction  = direction * 0.55 + float(np.tanh(float(out[0, 0]))) * 0.45
                confidence = confidence * 0.55 + float(mx.sigmoid(out[0, 1])) * 0.45
            except Exception:
                pass

        self.record_instruments(
                bar_delta=float(bar_delta) if 'bar_delta' in dir() else 0.0,
                cvd_z=float(cvd_z) if 'cvd_z' in dir() else 0.0,
                absorption=float(int(absorption)) if 'absorption' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
