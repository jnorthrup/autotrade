"""
Codec 05: Pairs Trading
Statistical arbitrage on co-moving coin pairs.

Within the 64-bar return window we reconstruct two rolling price series
(split at the midpoint as a proxy for two correlated instruments) and
compute their spread z-score. Entry when |z| > threshold, direction
against the z-score (mean reversion).

The rolling half-life gate ensures we only trade when the spread is
mean-reverting (estimated AR(1) beta < 0).
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


def _ar1_beta(y: np.ndarray) -> float:
    if len(y) < 4:
        return 0.0
    delta = y[1:] - y[:-1]
    y_lag = y[:-1]
    cov = float(np.cov(delta, y_lag)[0, 1])
    var = float(np.var(y_lag))
    return cov / (var + 1e-8)


class Codec05(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'pairs_trading'
        super().__init__(config)

        self.entry_z     = config.get('entry_z', 1.5)
        self.window      = config.get('window', 30)

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            self.model = None

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        returns = features[:min(len(features), 64)]
        n = len(returns)

        if n < self.window:
            return self.validate_signal(0.1, 0.0)

        price = float(market_data.get('price', 1.0))
        closes, highs, lows, volumes = self.get_ohlcv(market_data, features)

        prices = closes  # calibrated to pandas parquet data

        # Split the price series into two pseudo-legs:
        # leg_a = slow-EMA of prices (alpha=0.05)
        # leg_b = fast-EMA of prices (alpha=0.15)
        # The spread leg_b - leg_a approximates a pair spread
        alpha_f, alpha_s = 0.15, 0.05
        leg_a = np.zeros(n)
        leg_b = np.zeros(n)
        leg_a[0] = leg_b[0] = prices[0]
        for i in range(1, n):
            leg_a[i] = alpha_s * prices[i] + (1 - alpha_s) * leg_a[i-1]
            leg_b[i] = alpha_f * prices[i] + (1 - alpha_f) * leg_b[i-1]

        spread = leg_b - leg_a
        seg = spread[-self.window:]

        mu  = seg.mean()
        std = seg.std()
        z   = float((spread[-1] - mu) / (std + 1e-8))

        # Only trade when the spread is mean-reverting
        beta = _ar1_beta(seg)
        reverting = beta < -0.05

        if abs(z) < self.entry_z or not reverting:
            direction, confidence = 0.0, 0.15
        else:
            direction  = -float(np.sign(z))
            half_life  = float(-np.log(2) / (beta + 1e-10)) if beta < 0 else 999.0
            hl_factor  = min(1.0, 20.0 / (half_life + 1))   # shorter HL → higher conviction
            confidence = min(1.0, (abs(z) - self.entry_z) / 2.0 * hl_factor + 0.3)

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                direction  = direction * 0.6 + float(np.tanh(float(out[0, 0]))) * 0.4
                confidence = confidence * 0.6 + float(mx.sigmoid(out[0, 1])) * 0.4
            except Exception:
                pass

        self.record_instruments(
                spread_z=float(z) if 'z' in dir() else 0.0,
                ar1_beta=float(beta) if 'beta' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
