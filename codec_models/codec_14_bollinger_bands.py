"""
Codec 14: Bollinger Bands
Bollinger Band mean reversion and breakout.
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


class Codec14(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'bollinger_bands'
        super().__init__(config)
        
        self.bb_threshold = config.get('bb_threshold', 1.0)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price = market_data.get('price', 0)
        bb_upper = market_data.get('bb_upper', price)
        bb_lower = market_data.get('bb_lower', price)
        bb_mid = market_data.get('bb_mid', price)
        
        direction = 0.0
        confidence = 0.2
        
        if bb_upper > bb_lower:
            bb_width = bb_upper - bb_lower
            bb_position = (price - bb_lower) / bb_width if bb_width > 0 else 0.5
            
            if bb_position < 0:
                direction = 0.8
                confidence = min(abs(bb_position) + 0.4, 1.0)
            elif bb_position > 1:
                direction = -0.8
                confidence = min(bb_position - 1 + 0.4, 1.0)
            elif bb_position < 0.2:
                direction = 0.5
                confidence = 0.5
            elif bb_position > 0.8:
                direction = -0.5
                confidence = 0.5
        
        sma_20 = market_data.get('sma_15', price)
        bandwidth = (bb_upper - bb_lower) / sma_20 if sma_20 > 0 else 0
        if bandwidth < 0.02:
            confidence *= 0.5
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.5 + float(np.tanh(output[0, 1])) * 0.5
            except:
                pass
        
        self.record_instruments(
                bb_pct=float(pct_b) if 'pct_b' in dir() else 0.5,
                bb_width=float(bandwidth) if 'bandwidth' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

