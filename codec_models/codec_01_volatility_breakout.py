"""
Codec 01: Volatility Breakout
Captures explosive moves when volatility expands beyond normal ranges.
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


class Codec01(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'volatility_breakout'
        super().__init__(config)
        
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.lookback = config.get('lookback', 20)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
        else:
            self.model = None
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price = market_data.get('price', 0)
        high = market_data.get('high', price)
        low = market_data.get('low', price)
        atr = market_data.get('atr_14', (high - low) * 0.5)
        
        volatility_signal = 0.0
        if atr > 0:
            range_expansion = (high - low) / atr if atr > 0 else 0
            if range_expansion > self.atr_multiplier:
                momentum = market_data.get('momentum', 0)
                volatility_signal = np.sign(momentum) * min(range_expansion / 3, 1.0)
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                ml_confidence = float(mx.sigmoid(output[0, 0]))
                ml_direction = float(np.tanh(output[0, 1]))
                confidence = ml_confidence * 0.7 + abs(volatility_signal) * 0.3
                direction = ml_direction * 0.5 + volatility_signal * 0.5
            except:
                confidence = abs(volatility_signal) + 0.3
                direction = volatility_signal
        else:
            confidence = abs(volatility_signal) + 0.3
            direction = volatility_signal
        
        self.record_instruments(
                volatility_signal=float(volatility_signal),
                atr_norm=float(market_data.get('atr_14', 0.0)) / (float(market_data.get('price', 1.0)) + 1e-8),
                momentum=float(market_data.get('momentum', 0.0)),
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        if not HAS_MLX or self.model is None:
            return
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

