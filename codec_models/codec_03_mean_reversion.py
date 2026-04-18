"""
Codec 03: Mean Reversion
Bets on price returning to mean after extreme moves.
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


class Codec03(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'mean_reversion'
        super().__init__(config)
        
        self.lookback = config.get('lookback', 20)
        self.z_threshold = config.get('z_threshold', 2.0)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 3)
            )
        else:
            self.model = None
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price = market_data.get('price', 0)
        sma_20 = market_data.get('sma_15', price)
        rolling_std = market_data.get('vol_5m', price * 0.02)
        
        reversion_signal = 0.0
        if sma_20 > 0 and rolling_std > 0:
            z_score = (price - sma_20) / rolling_std
            if abs(z_score) > self.z_threshold:
                reversion_signal = -np.sign(z_score) * min(abs(z_score) / 4, 1.0)
        
        rsi = market_data.get('rsi_14', 50)
        rsi_signal = 0.0
        if rsi > 70:
            rsi_signal = -0.5
        elif rsi < 30:
            rsi_signal = 0.5
        
        combined = reversion_signal * 0.6 + rsi_signal * 0.4
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                ml_confidence = float(mx.sigmoid(output[0, 0]))
                ml_direction = float(np.tanh(output[0, 1]))
                confidence = ml_confidence * 0.5 + abs(combined) * 0.5
                direction = ml_direction * 0.3 + combined * 0.7
            except:
                confidence = abs(combined) + 0.2
                direction = combined
        else:
            confidence = abs(combined) + 0.2
            direction = combined
        
        self.record_instruments(
                returns_last=float(features[0]) if len(features) > 0 else 0.0,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

