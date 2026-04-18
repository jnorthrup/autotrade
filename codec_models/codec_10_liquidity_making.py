"""
Codec 10: Liquidity Making
Market-making style signal: exploit bid-ask spread dynamics and
volume imbalance without a live LOB feed.

Derived entirely from OHLCV:
  - Spread proxy = (high - low) / close  (Garman-Klass range normalised)
  - Garman-Klass volatility estimator (low-bias)
  - Roll's implicit spread estimator: -2 * sqrt(-cov(ret_t, ret_{t-1}))
    Positive cov → trending; negative → bid-ask bounce
  - Trade direction: fade large bar ranges when serial covariance is negative
    (i.e. price bouncing off bid/ask walls)
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


def _rolls_spread(returns: np.ndarray) -> float:
    """Roll (1984) implicit bid-ask spread estimator."""
    if len(returns) < 4:
        return 0.0
    cov = float(np.cov(returns[1:], returns[:-1])[0, 1])
    if cov >= 0:
        return 0.0
    return float(2.0 * np.sqrt(-cov))


def _garman_klass_vol(high_arr, low_arr, close_arr) -> float:
    """Garman-Klass low-bias volatility estimator."""
    if len(close_arr) < 2:
        return 0.0
    ln_hl = (np.log(high_arr / (low_arr + 1e-8))) ** 2
    ln_cc = (np.log(close_arr[1:] / (close_arr[:-1] + 1e-8))) ** 2
    n = min(len(ln_hl), len(ln_cc))
    gk = float(0.5 * np.mean(ln_hl[-n:]) - (2 * np.log(2) - 1) * np.mean(ln_cc[-n:]))
    return float(np.sqrt(max(gk, 0.0)))


class Codec10(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'liquidity_making'
        super().__init__(config)

        self.window = config.get('window', 20)

        # Stateful price history for Garman-Klass
        self._highs:  list = []
        self._lows:   list = []
        self._closes: list = []

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            self.model = None

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price = float(market_data.get('price', 1.0))
        high  = float(market_data.get('high', price))
        low   = float(market_data.get('low', price))

        self._highs.append(high);   self._highs  = self._highs[-self.window:]
        self._lows.append(low);     self._lows   = self._lows[-self.window:]
        self._closes.append(price); self._closes = self._closes[-self.window:]

        returns = features[:min(len(features), 64)]
        n = len(returns)

        if n < 4 or len(self._closes) < 4:
            return self.validate_signal(0.15, 0.0)

        # ── Roll's spread → sign of serial covariance ──────────────────────
        roll_spread = _rolls_spread(returns[-min(n, self.window):])
        cov = float(np.cov(returns[-self.window:][1:],
                           returns[-self.window:][:-1])[0, 1]) if n >= 4 else 0.0
        bouncing = cov < -1e-6   # negative serial cov = bid-ask bounce

        # ── Garman-Klass volatility ────────────────────────────────────────
        gk_vol = _garman_klass_vol(
            np.array(self._highs),
            np.array(self._lows),
            np.array(self._closes)
        )

        # ── Spread proxy (normalised bar range) ────────────────────────────
        bar_range     = (high - low) / (price + 1e-8)
        spread_proxy  = bar_range * 0.5   # half-spread proxy

        # ── Signal ─────────────────────────────────────────────────────────
        recent_ret = float(returns[-1]) if n >= 1 else 0.0

        if bouncing:
            # Fade last bar's direction (mean-revert off the spread)
            direction  = -float(np.sign(recent_ret)) if recent_ret != 0 else 0.0
            confidence = min(1.0, roll_spread * 20 + 0.3)
        else:
            # Trending orderflow — small conviction follow
            direction  = float(np.sign(recent_ret)) if abs(recent_ret) > gk_vol else 0.0
            confidence = min(1.0, abs(recent_ret) / (gk_vol + 1e-8) * 0.3 + 0.2)

        # Penalise high-vol regimes (dangerous for LM)
        vol_penalty = min(0.4, gk_vol * 20)
        confidence  = max(0.1, confidence - vol_penalty)

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                direction  = direction * 0.6 + float(np.tanh(float(out[0, 0]))) * 0.4
                confidence = confidence * 0.6 + float(mx.sigmoid(out[0, 1])) * 0.4
            except Exception:
                pass

        self.record_instruments(
                roll_spread=float(roll_spread) if 'roll_spread' in dir() else 0.0,
                gk_vol=float(gk_vol) if 'gk_vol' in dir() else 0.0,
                bouncing=float(int(bouncing)) if 'bouncing' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
