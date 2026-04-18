"""
Codec 15: MACD Crossover
Signal-line and histogram divergence detection.
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


class Codec15(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'macd_crossover'
        super().__init__(config)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        macd = market_data.get('macd', 0)
        macd_signal = market_data.get('macd_signal', 0)
        macd_hist = market_data.get('macd_hist', macd - macd_signal)
        
        direction = 0.0
        confidence = 0.2
        
        if macd_hist > 0:
            direction = min(macd_hist * 20, 1.0)
            confidence = min(abs(macd_hist) * 10 + 0.3, 1.0)
        elif macd_hist < 0:
            direction = max(macd_hist * 20, -1.0)
            confidence = min(abs(macd_hist) * 10 + 0.3, 1.0)
        
        momentum = market_data.get('momentum', 0)
        if np.sign(direction) == np.sign(momentum):
            confidence *= 1.2
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.5 + float(np.tanh(output[0, 1])) * 0.5
            except:
                pass
        
        self.record_instruments(
                macd_hist=float(macd_hist) if 'macd_hist' in dir() else 0.0,
                macd_line=float(macd_line) if 'macd_line' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

