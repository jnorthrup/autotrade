"""
Codec 02: Momentum Trend
Follows established trends with momentum confirmation.
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


class Codec02(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'momentum_trend'
        super().__init__(config)
        
        self.ema_fast = config.get('ema_fast', 10)
        self.ema_slow = config.get('ema_slow', 30)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
        else:
            self.model = None
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        ema_10 = market_data.get('ema_5', market_data.get('price', 0))
        ema_30 = market_data.get('ema_15', market_data.get('price', 0))
        ema_60 = market_data.get('ema_60', market_data.get('price', 0))
        
        trend_signal = 0.0
        if ema_30 > 0:
            trend_alignment = 0
            if ema_10 > ema_30:
                trend_alignment += 1
            else:
                trend_alignment -= 1
            if ema_30 > ema_60:
                trend_alignment += 1
            else:
                trend_alignment -= 1
            
            momentum = market_data.get('momentum', 0)
            trend_signal = (trend_alignment / 2) * 0.7 + np.sign(momentum) * 0.3
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                ml_confidence = float(mx.sigmoid(output[0, 0]))
                ml_direction = float(np.tanh(output[0, 1]))
                confidence = ml_confidence * 0.6 + abs(trend_signal) * 0.4
                direction = ml_direction * 0.4 + trend_signal * 0.6
            except:
                confidence = abs(trend_signal) + 0.25
                direction = trend_signal
        else:
            confidence = abs(trend_signal) + 0.25
            direction = trend_signal
        
        self.record_instruments(
                momentum_fast=float(market_data.get('momentum', 0.0)),
                returns_last=float(features[0]) if len(features) > 0 else 0.0,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

