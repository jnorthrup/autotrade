"""
Codec 21: Random Forest Classifier
Ensemble of independent decision-tree-style classifiers operating on
engineered indicator features, without any stochastic randomness in inference.
Each "tree" applies a fixed deterministic split sequence on a distinct
feature subspace; their votes are majority-aggregated.
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


def _ema(prices: np.ndarray, span: int) -> float:
    """Exponential moving average of a price series."""
    if len(prices) == 0:
        return 0.0
    alpha = 2.0 / (span + 1)
    val = float(prices[0])
    for p in prices[1:]:
        val = alpha * float(p) + (1 - alpha) * val
    return val


def _rsi(returns: np.ndarray, period: int = 14) -> float:
    """RSI from return series."""
    if len(returns) < period:
        return 50.0
    r = returns[-period:]
    gains = r[r > 0].mean() if (r > 0).any() else 0.0
    losses = -r[r < 0].mean() if (r < 0).any() else 0.0
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    h, l, c = highs[-period:], lows[-period:], closes[-period:]
    prev_c = closes[-(period + 1):-1] if len(closes) > period else closes[:-1]
    tr = np.maximum(h[-len(prev_c):] - l[-len(prev_c):],
                    np.maximum(np.abs(h[-len(prev_c):] - prev_c),
                               np.abs(l[-len(prev_c):] - prev_c)))
    return float(tr.mean()) if len(tr) > 0 else 0.0


class _DecisionTree:
    """A hand-coded decision tree operating on a specific feature subspace."""

    def __init__(self, feature_idx: int, thresholds: list, directions: list):
        self.feature_idx = feature_idx
        self.thresholds = thresholds   # ascending split points
        self.directions = directions   # direction vote per leaf

    def predict(self, features: np.ndarray) -> float:
        val = float(features[self.feature_idx]) if len(features) > self.feature_idx else 0.0
        for i, thresh in enumerate(self.thresholds):
            if val <= thresh:
                return float(self.directions[i])
        return float(self.directions[-1])


class Codec21(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'random_forest_classifier'
        super().__init__(config)

        self.n_trees = config.get('n_trees', 20)
        self.lookback = config.get('lookback', 30)

        # Pre-built deterministic forest on 8 engineered features:
        # 0: momentum_5   1: momentum_20   2: rsi_14       3: ema_ratio_5_20
        # 4: vol_20       5: atr_ratio     6: close_z20    7: volume_ratio
        self._trees = [
            _DecisionTree(0, [-0.02, 0.0, 0.02],  [-1, -0.5, 0.5, 1]),
            _DecisionTree(1, [-0.01, 0.0, 0.01],  [-1, -0.3, 0.3, 1]),
            _DecisionTree(2, [30, 45, 55, 70],    [-1, -0.5, 0.0, 0.5, 1]),
            _DecisionTree(3, [0.98, 1.0, 1.02],   [-1, -0.3, 0.3, 1]),
            _DecisionTree(4, [0.005, 0.015, 0.03], [0.2, -0.2, -0.5, -1]),
            _DecisionTree(5, [0.5, 1.0, 2.0],     [0, -0.3, -0.6, -1]),
            _DecisionTree(6, [-2.0, -0.5, 0.5, 2.0], [-1, -0.3, 0.3, 1]),
            _DecisionTree(7, [0.7, 1.0, 1.5],     [-0.3, 0.0, 0.3, 1]),
        ]
        # Pad to n_trees by cycling through the base trees
        self._forest = [self._trees[i % len(self._trees)] for i in range(self.n_trees)]

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

    def _build_features(self, market_data: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        """Build the 8-feature engineered vector from OHLCV context."""
        price  = float(market_data.get('price', 1.0))
        high   = float(market_data.get('high', price))
        low    = float(market_data.get('low', price))
        volume = float(market_data.get('volume', 1.0))

        # Use real pandas-sourced OHLCV arrays via get_ohlcv()
        prices_proxy, highs_arr, lows_arr, volumes_arr = self.get_ohlcv(market_data, features)
        returns = market_data.get('ret_series', features[:min(len(features), 64)])
        n = len(prices_proxy)
        p5  = float(np.mean(prices_proxy[-5:]))  if n >= 5  else price
        p20 = float(np.mean(prices_proxy[-20:])) if n >= 20 else price
        mom5  = (price - p5)  / (p5  + 1e-8)
        mom20 = (price - p20) / (p20 + 1e-8)

        rsi14 = _rsi(returns)

        ema5  = _ema(prices_proxy[-20:], 5)  if n >= 5  else price
        ema20 = _ema(prices_proxy[-20:], 20) if n >= 20 else price
        ema_ratio = ema5 / (ema20 + 1e-8)

        vol20 = float(np.std(returns[-20:])) if n >= 20 else 0.01

        atr14 = _atr(
            np.full(min(n, 14), high),
            np.full(min(n, 14), low),
            prices_proxy[-min(n, 15):]
        )
        atr_ratio = atr14 / (price + 1e-8)

        mean20 = float(np.mean(prices_proxy[-20:])) if n >= 20 else price
        std20  = float(np.std(prices_proxy[-20:]))  if n >= 20 else 1.0
        close_z = (price - mean20) / (std20 + 1e-8)

        vol_proxy = float(market_data.get('volume', 1.0))
        mean_vol = float(np.mean(np.abs(returns[-20:]))) if n >= 20 else 1.0
        volume_ratio = vol_proxy / (mean_vol * 1e6 + 1e-8)  # normalised

        return np.array([mom5, mom20, rsi14, ema_ratio, vol20, atr_ratio, close_z, volume_ratio],
                        dtype=np.float32)

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        eng = self._build_features(market_data, features)

        # Majority vote across forest
        votes = np.array([tree.predict(eng) for tree in self._forest])
        raw_direction = float(np.mean(votes))
        vote_agreement = float(np.abs(raw_direction))  # 0..1

        direction  = np.sign(raw_direction) if vote_agreement > 0.1 else 0.0
        confidence = min(1.0, vote_agreement * 1.2 + 0.2)

        # Blend with MLX model if available
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
                rf_vote_agreement=float(vote_agreement) if 'vote_agreement' in dir() else 0.0,
                rf_raw_direction=float(raw_direction) if 'raw_direction' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
