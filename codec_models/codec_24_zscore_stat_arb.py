"""
Codec 24: Z-Score Stat Arb
Multi-asset z-score mean-reversion episodes.

Signal logic:
  1. Compute rolling z-scores at three lookback windows (10, 20, 60 bars)
  2. Cointegration-style spread via price deviation from its own EMA channel
  3. Cross-sectional rank via percentile within the observed price distribution
  4. Weighted combination → mean-reversion direction + conviction proportional
     to |z| when |z| > entry_threshold
  5. Half-life estimation from AR(1) fit to detect reverting vs trending regime
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


def _ema(arr: np.ndarray, span: int) -> float:
    alpha = 2.0 / (span + 1)
    v = float(arr[0])
    for x in arr[1:]:
        v = alpha * float(x) + (1 - alpha) * v
    return v


def _rolling_zscore(prices: np.ndarray, window: int) -> float:
    """Z-score of the last price vs rolling mean/std over `window` bars."""
    if len(prices) < window:
        window = max(2, len(prices))
    seg = prices[-window:]
    mu  = seg.mean()
    std = seg.std()
    return float((prices[-1] - mu) / (std + 1e-8))


def _half_life(prices: np.ndarray) -> float:
    """
    Estimate mean-reversion half-life via AR(1) fit:
        Δy_t = β * y_{t-1} + ε  →  half_life = -log(2) / β
    Returns inf if the series is trending (β >= 0).
    """
    if len(prices) < 10:
        return float('inf')
    y   = prices[1:]
    y_1 = prices[:-1]
    # OLS: β = cov(Δy, y_{t-1}) / var(y_{t-1})
    delta = y - y_1
    cov = float(np.cov(delta, y_1)[0, 1])
    var = float(np.var(y_1))
    if var < 1e-10:
        return float('inf')
    beta = cov / var
    if beta >= 0:
        return float('inf')
    return float(-np.log(2) / beta)


def _ema_channel_spread(prices: np.ndarray, fast: int = 10, slow: int = 30) -> float:
    """
    Deviation of current price from the EMA midline as a fraction of the
    EMA channel width (fast−slow spread).  Returns value in (-1, 1).
    """
    if len(prices) < slow:
        return 0.0
    e_fast = _ema(prices[-fast:], fast)
    e_slow = _ema(prices[-slow:], slow)
    midline = (e_fast + e_slow) / 2.0
    channel = abs(e_fast - e_slow) + 1e-8
    return float((prices[-1] - midline) / channel)


class Codec24(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'zscore_stat_arb'
        super().__init__(config)

        self.entry_z       = config.get('entry_z', 1.5)       # |z| threshold to open
        self.exit_z        = config.get('exit_z', 0.5)        # |z| threshold to close (not used here but stored)
        self.max_half_life = config.get('max_half_life', 40)   # reject trending regimes
        self.lookbacks     = config.get('lookbacks', [10, 20, 60])
        self.weights       = config.get('weights', [0.5, 0.3, 0.2])

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            self.model = None

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price = float(market_data.get('price', 1.0))

        # Reconstruct price series from return vector in features
        returns = features[:min(len(features), 64)]
        n = len(returns)

        if n < max(self.lookbacks):
            # Not enough history — emit weak neutral
            return self.validate_signal(0.1, 0.0)

        closes, highs, lows, volumes = self.get_ohlcv(market_data, features)


        prices = closes  # calibrated to pandas parquet data

        # ── 1. Rolling z-scores at multiple lookbacks ─────────────────────
        zscores = [_rolling_zscore(prices, lb) for lb in self.lookbacks]
        # Weighted composite z-score (mean-reversion: fade the sign)
        composite_z = float(np.dot(zscores, self.weights))

        # ── 2. EMA channel spread ─────────────────────────────────────────
        channel_spread = _ema_channel_spread(prices, fast=10, slow=30)

        # ── 3. Half-life filter ───────────────────────────────────────────
        hl = _half_life(prices[-60:] if n >= 60 else prices)
        reverting = hl < self.max_half_life   # True → mean-reverting regime

        # ── 4. Cross-sectional percentile rank ────────────────────────────
        # Where does the current price sit within its own distribution?
        pct_rank = float(np.mean(prices < price))   # 0 = bottom, 1 = top

        # ── 5. Entry gate & signal composition ───────────────────────────
        total_z = composite_z * 0.6 + channel_spread * 0.4

        if abs(total_z) < self.entry_z:
            # Below entry threshold — flat, low conviction
            direction  = 0.0
            confidence = 0.15
        else:
            # Mean-reversion: fade the z-score direction
            direction = -float(np.sign(total_z))

            # Conviction proportional to |z| capped at 1.0
            raw_conviction = min(1.0, (abs(total_z) - self.entry_z) / 2.0 + 0.3)

            # Downweight when the regime is trending (not mean-reverting)
            regime_factor = 0.7 if reverting else 0.35

            # Percentile extremes add conviction (overextended → stronger fade)
            pct_factor = 1.0 + 0.3 * (abs(pct_rank - 0.5) * 2)   # 1.0–1.3

            confidence = min(1.0, raw_conviction * regime_factor * pct_factor)

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                ml_dir  = float(np.tanh(float(out[0, 0])))
                ml_conf = float(mx.sigmoid(out[0, 1]))
                direction  = direction * 0.6 + ml_dir * 0.4
                confidence = confidence * 0.6 + ml_conf * 0.4
            except Exception:
                pass

        self.record_instruments(
                composite_z=float(composite_z) if 'composite_z' in dir() else 0.0,
                channel_spread=float(channel_spread) if 'channel_spread' in dir() else 0.0,
                half_life=float(min(hl, 999.0)) if 'hl' in dir() else 999.0,
                pct_rank=float(pct_rank) if 'pct_rank' in dir() else 0.5,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
