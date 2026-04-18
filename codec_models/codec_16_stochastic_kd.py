"""
Codec 16: Stochastic KD
%K/%D crossover with overbought/oversold filter.
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


class Codec16(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'stochastic_kd'
        super().__init__(config)
        
        self.oversold = config.get('oversold', 20)
        self.overbought = config.get('overbought', 80)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        stoch_k = market_data.get('stoch_k', 50)
        stoch_d = market_data.get('stoch_d', 50)
        
        direction = 0.0
        confidence = 0.2
        
        cross_signal = stoch_k - stoch_d
        
        if stoch_k < self.oversold and stoch_d < self.oversold:
            if cross_signal > 0:
                direction = min(cross_signal / 10 + 0.5, 1.0)
                confidence = 0.7 + (self.oversold - stoch_k) / 100
        elif stoch_k > self.overbought and stoch_d > self.overbought:
            if cross_signal < 0:
                direction = max(cross_signal / 10 - 0.5, -1.0)
                confidence = 0.7 + (stoch_k - self.overbought) / 100
        else:
            direction = np.sign(cross_signal) * min(abs(cross_signal) / 10, 0.5)
            confidence = 0.4
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.5 + float(np.tanh(output[0, 1])) * 0.5
            except:
                pass
        
        self.record_instruments(
                stoch_k=float(k_pct) if 'k_pct' in dir() else 50.0,
                stoch_d=float(d_pct) if 'd_pct' in dir() else 50.0,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

