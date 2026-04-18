"""
Codec 13: RSI Reversal
RSI-based mean reversion strategy.
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


class Codec13(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'rsi_reversal'
        super().__init__(config)
        
        self.oversold = config.get('oversold', 30)
        self.overbought = config.get('overbought', 70)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        rsi = market_data.get('rsi_14', 50)
        
        direction = 0.0
        confidence = 0.2
        
        if rsi < self.oversold:
            direction = (self.oversold - rsi) / self.oversold
            confidence = min(1.0 - rsi / self.oversold + 0.3, 1.0)
        elif rsi > self.overbought:
            direction = -(rsi - self.overbought) / (100 - self.overbought)
            confidence = min((rsi - self.overbought) / (100 - self.overbought) + 0.3, 1.0)
        
        stoch = market_data.get('stochastic', 50)
        if stoch < 20 and rsi < self.oversold:
            direction *= 1.3
            confidence = min(confidence * 1.2, 1.0)
        elif stoch > 80 and rsi > self.overbought:
            direction *= 1.3
            confidence = min(confidence * 1.2, 1.0)
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.6 + float(np.tanh(output[0, 1])) * 0.4
            except:
                pass
        
        self.record_instruments(
                rsi_14=float(rsi) if 'rsi' in dir() else 50.0,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

