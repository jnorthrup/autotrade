"""
Codec 12: Composite Alpha
Aggregates signals from multiple independent alpha sources computed
directly inside this codec — unlike the HRM which combines codec outputs,
this codec runs its own internal mini-ensemble from raw OHLCV.

Internal alpha sources:
  1. Momentum crossover (EMA_fast vs EMA_slow)
  2. RSI mean-reversion gate
  3. Bollinger-band squeeze breakout
  4. Price channel breakout (Donchian)
  5. Volume momentum (return × sqrt(normalised volume))

Each source produces (signal ∈ [-1,+1], quality ∈ [0,1]).
Final signal = quality-weighted average; confidence = portfolio IC proxy.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
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


def _rsi(returns: np.ndarray, period: int = 14) -> float:
    if len(returns) < period:
        return 50.0
    r = returns[-period:]
    gains = r[r > 0].mean() if (r > 0).any() else 0.0
    losses = -r[r < 0].mean() if (r < 0).any() else 1e-8
    return 100.0 - 100.0 / (1.0 + gains / losses)


class Codec12(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'composite_alpha'
        super().__init__(config)

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            self.model = None

    # ── Internal alpha sources ─────────────────────────────────────────────

    def _alpha_momentum(self, prices: np.ndarray, returns: np.ndarray) -> Tuple[float, float]:
        n = len(prices)
        if n < 26:
            return 0.0, 0.0
        ema_f = _ema(prices, 12)
        ema_s = _ema(prices, 26)
        cross = (ema_f - ema_s) / (ema_s + 1e-8)
        sig   = float(np.tanh(cross * 50))
        qual  = min(1.0, abs(cross) * 100 + 0.2)
        return sig, qual

    def _alpha_rsi(self, returns: np.ndarray) -> Tuple[float, float]:
        rsi14 = _rsi(returns)
        if rsi14 < 30:
            return 1.0, (30 - rsi14) / 30.0
        elif rsi14 > 70:
            return -1.0, (rsi14 - 70) / 30.0
        return 0.0, 0.1

    def _alpha_bollinger(self, prices: np.ndarray) -> Tuple[float, float]:
        n = len(prices)
        w = min(n, 20)
        seg  = prices[-w:]
        mu   = seg.mean()
        std  = seg.std() + 1e-8
        bb   = (prices[-1] - mu) / (2.0 * std)   # -1 at lower, +1 at upper
        if abs(bb) < 0.7:
            return 0.0, 0.1
        # Breakout: follow direction; squeeze: fade if bb > 1
        sig  = float(np.sign(bb)) if abs(bb) > 1.0 else -float(np.sign(bb))
        qual = min(1.0, (abs(bb) - 0.7) * 0.8 + 0.2)
        return sig, qual

    def _alpha_donchian(self, prices: np.ndarray, window: int = 20) -> Tuple[float, float]:
        n = len(prices)
        if n < window:
            return 0.0, 0.0
        ch_max = prices[-window:].max()
        ch_min = prices[-window:].min()
        ch_rng = ch_max - ch_min + 1e-8
        pos = (prices[-1] - ch_min) / ch_rng   # 0..1
        sig  = 2.0 * pos - 1.0                 # -1 at lower, +1 at upper
        # High breakout: follow; mid-channel: low conviction
        qual = min(1.0, abs(pos - 0.5) * 2.5 * 0.8 + 0.1)
        return float(sig), qual

    def _alpha_volume_momentum(self, returns: np.ndarray, price: float, volume: float) -> Tuple[float, float]:
        ret1 = float(returns[-1]) if len(returns) >= 1 else 0.0
        hist_vol = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.01
        vol_norm = volume / (hist_vol * 1e6 + 1e-8)   # rough normalisation
        sgnl  = float(np.sign(ret1)) * min(1.0, np.sqrt(vol_norm) * 0.5)
        qual  = min(1.0, abs(ret1) / (hist_vol + 1e-8) * 0.3 + 0.15)
        return sgnl, qual

    # ── Main forward ──────────────────────────────────────────────────────

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price   = float(market_data.get('price', 1.0))
        volume  = float(market_data.get('volume', 1.0))
        returns = features[:min(len(features), 64)]
        n = len(returns)

        if n < 5:
            return self.validate_signal(0.2, 0.0)

        closes, highs, lows, volumes = self.get_ohlcv(market_data, features)


        prices = closes  # calibrated to pandas parquet data

        alphas: List[Tuple[float, float]] = [
            self._alpha_momentum(prices, returns),
            self._alpha_rsi(returns),
            self._alpha_bollinger(prices),
            self._alpha_donchian(prices),
            self._alpha_volume_momentum(returns, price, volume),
        ]

        total_qual = sum(q for _, q in alphas)
        if total_qual < 1e-6:
            return self.validate_signal(0.2, 0.0)

        direction  = float(sum(s * q for s, q in alphas) / total_qual)
        # IC proxy: how much do the sources agree?
        sigs = np.array([s for s, _ in alphas])
        agreement = float(abs(np.mean(np.sign(sigs[sigs != 0])))) if (sigs != 0).any() else 0.0
        confidence = min(1.0, agreement * 0.5 + abs(direction) * 0.3 + 0.2)

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                direction  = direction * 0.5 + float(np.tanh(float(out[0, 0]))) * 0.5
                confidence = confidence * 0.5 + float(mx.sigmoid(out[0, 1])) * 0.5
            except Exception:
                pass

        self.record_instruments(
                alpha_count=float(len([s for s,_ in alphas if s != 0.0])) if 'alphas' in dir() else 0.0,
                alpha_agreement=float(agreement) if 'agreement' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
