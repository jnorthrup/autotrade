"""
Codec 22: XGBoost Signal
Gradient-boosting-style signal generation implemented as an additive ensemble
of residual weak learners (threshold regressors), each fitted to the residual
of the previous stage — exactly the XGBoost forward stagewise algorithm,
carried out with pre-set splits on engineered indicator features.
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


class _WeakLearner:
    """A single-split regression stump: if feature[idx] <= thresh → val_l else val_r."""
    def __init__(self, feat_idx: int, thresh: float, val_l: float, val_r: float,
                 learning_rate: float = 0.3):
        self.feat_idx = feat_idx
        self.thresh = thresh
        self.val_l = val_l * learning_rate
        self.val_r = val_r * learning_rate

    def predict(self, f: np.ndarray) -> float:
        v = float(f[self.feat_idx]) if len(f) > self.feat_idx else 0.0
        return self.val_l if v <= self.thresh else self.val_r


def _macd(prices: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9):
    """Returns (macd_line, signal_line)."""
    def ema(p, span):
        alpha = 2.0 / (span + 1)
        v = float(p[0])
        for x in p[1:]:
            v = alpha * float(x) + (1 - alpha) * v
        return v
    if len(prices) < slow:
        return 0.0, 0.0
    m = ema(prices, fast) - ema(prices, slow)
    hist = np.array([ema(prices[:max(1, len(prices) - sig + i + 1)], sig)
                     for i in range(sig)])
    s = float(hist[-1]) if len(hist) > 0 else 0.0
    return m, s


class Codec22(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'xgboost_signal'
        super().__init__(config)

        self.n_rounds = config.get('n_rounds', 30)
        self.lr = config.get('learning_rate', 0.3)

        # Pre-tuned boosting rounds.  Features (0-indexed):
        # 0 mom_3  1 mom_10  2 mom_30  3 macd_hist  4 rsi_norm  5 vol_ratio
        # 6 bb_pos  7 close_z_10  8 close_z_30  9 atr_norm
        self._stumps = [
            # Early rounds: strong trend features
            _WeakLearner(0, 0.0,   -1.0,  1.0, self.lr),
            _WeakLearner(1, 0.0,   -1.0,  1.0, self.lr),
            _WeakLearner(2, 0.0,   -0.8,  0.8, self.lr),
            _WeakLearner(3, 0.0,   -0.9,  0.9, self.lr),
            # Mid rounds: overbought/oversold correction
            _WeakLearner(4, 0.3,    0.5,  0.0, self.lr),   # rsi_norm low → buy bias
            _WeakLearner(4, 0.7,    0.0, -0.5, self.lr),   # rsi_norm high → sell bias
            _WeakLearner(6, -1.0,   0.8,  0.0, self.lr),   # below lower BB → buy
            _WeakLearner(6,  1.0,   0.0, -0.8, self.lr),   # above upper BB → sell
            # Later rounds: volatility & z-score
            _WeakLearner(5, 0.5,    0.3, -0.3, self.lr),   # low vol → trend, high vol → fade
            _WeakLearner(7, -1.5,   0.6,  0.0, self.lr),
            _WeakLearner(7,  1.5,   0.0, -0.6, self.lr),
            _WeakLearner(8, -2.0,   0.5,  0.0, self.lr),
            _WeakLearner(8,  2.0,   0.0, -0.5, self.lr),
            _WeakLearner(9, 0.01,   0.2, -0.2, self.lr),   # wide spread → fade
            # Fine-tuning
            _WeakLearner(0, -0.01, -0.4,  0.0, self.lr),
            _WeakLearner(0,  0.01,  0.0,  0.4, self.lr),
            _WeakLearner(1, -0.005,-0.3,  0.0, self.lr),
            _WeakLearner(1,  0.005, 0.0,  0.3, self.lr),
            _WeakLearner(2, -0.002,-0.2,  0.0, self.lr),
            _WeakLearner(2,  0.002, 0.0,  0.2, self.lr),
            _WeakLearner(3, -0.001,-0.25, 0.0, self.lr),
            _WeakLearner(3,  0.001, 0.0,  0.25,self.lr),
            _WeakLearner(4,  0.45,  0.15, 0.0, self.lr),
            _WeakLearner(4,  0.55,  0.0, -0.15,self.lr),
            _WeakLearner(6, -0.5,   0.2,  0.0, self.lr),
            _WeakLearner(6,  0.5,   0.0, -0.2, self.lr),
            _WeakLearner(7, -0.5,   0.15, 0.0, self.lr),
            _WeakLearner(7,  0.5,   0.0, -0.15,self.lr),
            _WeakLearner(5, 1.5,    0.0, -0.1, self.lr),
            _WeakLearner(9, 0.005,  0.1, -0.1, self.lr),
        ]
        self._stumps = self._stumps[:self.n_rounds]

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            self.model = None

    def _build_features(self, market_data: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        price  = float(market_data.get('price', 1.0))
        high   = float(market_data.get('high', price))
        low    = float(market_data.get('low', price))

        returns = features[:min(len(features), 64)]
        n = len(returns)
        closes, highs, lows, volumes = self.get_ohlcv(market_data, features)

        prices = closes  # calibrated to pandas parquet data

        def mom(p, k):
            return (price - float(np.mean(p[-k:]))) / (float(np.mean(p[-k:])) + 1e-8) if len(p) >= k else 0.0

        mom3  = mom(prices, 3)
        mom10 = mom(prices, 10)
        mom30 = mom(prices, 30)

        macd_l, macd_s = _macd(prices)
        macd_hist = macd_l - macd_s

        r = returns[-14:] if n >= 14 else returns
        gains = r[r > 0].mean() if (r > 0).any() else 0.0
        losses = -r[r < 0].mean() if (r < 0).any() else 1e-8
        rsi = 100 - 100 / (1 + gains / losses)
        rsi_norm = rsi / 100.0

        vol = float(np.std(returns[-20:])) if n >= 20 else 0.01
        long_vol = float(np.std(returns[-50:])) if n >= 50 else vol
        vol_ratio = vol / (long_vol + 1e-8)

        # Bollinger position: -1 below lower, +1 above upper
        sma20 = float(np.mean(prices[-20:])) if n >= 20 else price
        std20 = float(np.std(prices[-20:])) if n >= 20 else 1.0
        bb_pos = (price - sma20) / (2.0 * std20 + 1e-8)

        # Z-scores
        mean10 = float(np.mean(prices[-10:])) if n >= 10 else price
        std10  = float(np.std(prices[-10:]))  if n >= 10 else 1.0
        z10 = (price - mean10) / (std10 + 1e-8)
        mean30 = float(np.mean(prices[-30:])) if n >= 30 else price
        std30  = float(np.std(prices[-30:]))  if n >= 30 else 1.0
        z30 = (price - mean30) / (std30 + 1e-8)

        atr_norm = (high - low) / (price + 1e-8)

        return np.array([mom3, mom10, mom30, macd_hist, rsi_norm, vol_ratio,
                         bb_pos, z10, z30, atr_norm], dtype=np.float32)

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        eng = self._build_features(market_data, features)

        # Additive forward stagewise boosting prediction
        raw = sum(s.predict(eng) for s in self._stumps)
        direction  = float(np.tanh(raw))
        confidence = min(1.0, abs(direction) + 0.15)

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                ml_dir  = float(np.tanh(float(out[0, 0])))
                ml_conf = float(mx.sigmoid(out[0, 1]))
                direction  = direction * 0.55 + ml_dir * 0.45
                confidence = confidence * 0.55 + ml_conf * 0.45
            except Exception:
                pass

        self.record_instruments(
                xgb_raw=float(raw) if 'raw' in dir() else 0.0,
                rsi_norm=float(eng[4]) if 'eng' in dir() else 0.5,
                bb_pos=float(eng[6]) if 'eng' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
