"""
Codec 09: Correlation Trading
Exploits inter-asset correlation breakdowns and restoration.

Without a live multi-asset feed, correlation is estimated from the bar-level
return auto-correlation structure (serial correlation = proxy for whether
the asset is trending WITH its historical self or reverting).

Signal logic:
  - Ljung-Box statistic on return lags 1–5: detects serial correlation
  - Positive autocorrelation → trend follow the recent direction
  - Negative autocorrelation → mean revert
  - Breadth indicator: consistency of signal direction across multiple
    rolling windows (5 / 10 / 20 / 40 bars) — acts as cross-timeframe
    correlation of the signal itself
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


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return 0.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:])) / denom


def _ljung_box_stat(returns: np.ndarray, max_lag: int = 5) -> Tuple[float, float]:
    """
    Simple Ljung-Box Q statistic for lags 1..max_lag.
    Returns (Q_stat, weighted_autocorr).
    Positive weighted_autocorr → trending; negative → reverting.
    """
    n = len(returns)
    if n < max_lag + 2:
        return 0.0, 0.0
    acfs = [_autocorr(returns, k) for k in range(1, max_lag + 1)]
    weights = np.array([1.0 / k for k in range(1, max_lag + 1)])
    weighted_acf = float(np.dot(acfs, weights) / weights.sum())
    q = float(n * (n + 2) * sum(a ** 2 / (n - k) for k, a in enumerate(acfs, 1)))
    return q, weighted_acf


class Codec09(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'correlation_trading'
        super().__init__(config)

        self.lookbacks  = config.get('lookbacks', [5, 10, 20, 40])

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

        if n < max(self.lookbacks):
            return self.validate_signal(0.15, 0.0)

        # ── Autocorrelation structure across multiple windows ──────────────
        signals = []
        for lb in self.lookbacks:
            if n < lb:
                continue
            seg = returns[-lb:]
            _, acf = _ljung_box_stat(seg, max_lag=min(5, lb // 2))
            recent_dir = float(np.sign(seg[-3:].mean())) if len(seg) >= 3 else 0.0
            # Trend-follow if autocorr positive, revert if negative
            if abs(acf) > 0.05:
                sig = float(np.sign(acf)) * recent_dir
                signals.append(sig)

        if not signals:
            return self.validate_signal(0.15, 0.0)

        # Breadth: agreement across timeframes
        direction   = float(np.mean(signals))
        breadth     = float(np.abs(np.mean(np.sign(signals))))  # 0..1
        confidence  = min(1.0, breadth * 0.6 + abs(direction) * 0.2 + 0.15)

        direction = float(np.clip(direction, -1.0, 1.0))

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                direction  = direction * 0.6 + float(np.tanh(float(out[0, 0]))) * 0.4
                confidence = confidence * 0.6 + float(mx.sigmoid(out[0, 1])) * 0.4
            except Exception:
                pass

        self.record_instruments(
                weighted_acf=float(sum(s for s, _ in ([(s, q) for s, q in [(float(np.mean(signals)), 1.0)] if signals] or [(0.0, 0.0)]))) if 'signals' in dir() and signals else 0.0,
                breadth=float(breadth) if 'breadth' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
