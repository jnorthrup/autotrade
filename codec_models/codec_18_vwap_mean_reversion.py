"""
Codec 18: VWAP Mean Reversion
Volume-weighted anchor reversion strategy.
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


class Codec18(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'vwap_mean_reversion'
        super().__init__(config)
        
        self.deviation_threshold = config.get('deviation_threshold', 0.01)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price = market_data.get('price', 0)
        vwap = market_data.get('vwap', price)
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume)
        
        direction = 0.0
        confidence = 0.2
        
        if vwap > 0:
            deviation = (price - vwap) / vwap
            
            if abs(deviation) > self.deviation_threshold:
                direction = -np.sign(deviation) * min(abs(deviation) * 20, 1.0)
                confidence = min(abs(deviation) * 30 + 0.3, 1.0)
            
            vol_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            if vol_ratio > 1.5:
                confidence = min(confidence * 1.2, 1.0)
        
        regime = market_data.get('regime_label', 1)
        if regime == 1:
            confidence *= 1.2
        else:
            confidence *= 0.7
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.5 + float(np.tanh(output[0, 1])) * 0.5
            except:
                pass
        
        self.record_instruments(
                vwap_dev=float(vwap_dev) if 'vwap_dev' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

