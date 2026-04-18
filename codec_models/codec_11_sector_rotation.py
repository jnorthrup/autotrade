"""
Codec 11: Sector Rotation
Rotates into the strongest recent performer and out of the weakest.

In a single-asset bar stream, "sector" is proxied by relative performance
across multiple rolling windows (analogous to sector strength vs market):
  - Momentum at 5 / 10 / 20 / 60 bar windows → rank each
  - Weight recent (5-bar) momentum highest; longer windows provide regime context
  - Sector score = weighted rank sum; long if strong, short if weak
  - Regime filter: only favour longs in uptrend regime, shorts in downtrend

Also incorporates a relative-value signal: how does this bar's return
compare against its own historical distribution (percentile rank).
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


class Codec11(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'sector_rotation'
        super().__init__(config)

        # Window / weight pairs — short windows weighted highest
        self.windows  = config.get('windows',  [5, 10, 20, 60])
        self.weights  = config.get('weights',  [0.40, 0.30, 0.20, 0.10])

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            self.model = None

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        returns = features[:min(len(features), 64)]
        n = len(returns)
        price = float(market_data.get('price', 1.0))

        if n < max(self.windows):
            return self.validate_signal(0.15, 0.0)

        closes, highs, lows, volumes = self.get_ohlcv(market_data, features)


        prices = closes  # calibrated to pandas parquet data

        # ── Momentum scores per window ─────────────────────────────────────
        mom_scores = []
        for w in self.windows:
            if n < w:
                mom_scores.append(0.0)
                continue
            ret_w  = float((prices[-1] - prices[-w]) / (prices[-w] + 1e-8))
            mom_scores.append(ret_w)

        # Weighted combination
        mom_arr   = np.array(mom_scores)
        composite = float(np.dot(mom_arr, self.weights[:len(mom_arr)]))

        # ── Relative-value: percentile rank of current 1-bar return ───────
        ret1  = float(returns[-1]) if n >= 1 else 0.0
        hist  = returns[-60:] if n >= 60 else returns
        pct   = float(np.mean(hist < ret1))  # 0..1

        # ── Regime context: 20-bar trend slope ────────────────────────────
        if n >= 20:
            x = np.arange(20, dtype=np.float32)
            y = prices[-20:]
            slope = float(np.polyfit(x, y, 1)[0])
            regime = float(np.sign(slope))
        else:
            regime = 0.0

        # ── Signal ─────────────────────────────────────────────────────────
        direction = float(np.sign(composite)) * (0.7 + 0.3 * abs(2 * pct - 1))
        if regime != 0.0:
            direction = direction * 0.7 + regime * 0.3

        confidence = min(1.0, abs(composite) * 10 + 0.25 + abs(2 * pct - 1) * 0.2)

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                direction  = direction * 0.6 + float(np.tanh(float(out[0, 0]))) * 0.4
                confidence = confidence * 0.6 + float(mx.sigmoid(out[0, 1])) * 0.4
            except Exception:
                pass

        self.record_instruments(
                composite_mom=float(composite) if 'composite' in dir() else 0.0,
                pct_rank=float(pct) if 'pct' in dir() else 0.5,
                regime=float(regime) if 'regime' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
