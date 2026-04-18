"""
Codec 17: ADX Trend Strength
Directional movement + ADX power filter.
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


class Codec17(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'adx_trend_strength'
        super().__init__(config)
        
        self.adx_threshold = config.get('adx_threshold', 25)
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        # InstrumentPanel emits 'adx', 'plus_di', 'minus_di'
        adx      = float(market_data.get('adx',      market_data.get('adx_14', 0.0)))
        plus_di  = float(market_data.get('plus_di',  0.0))
        minus_di = float(market_data.get('minus_di', 0.0))

        di_diff    = plus_di - minus_di
        direction  = float(np.tanh(di_diff / 20.0))   # DI spread → direction

        # Graded conviction: full above threshold, tapered below, never zero
        if adx >= self.adx_threshold:          # strong trend ≥ 25
            adx_factor = min(1.0, adx / 50.0 + 0.3)
        elif adx >= 15:                        # moderate trend 15-25
            adx_factor = 0.15 + (adx - 15) / 10.0 * 0.3
        else:                                  # weak / choppy, follow DI gently
            adx_factor = 0.1 + adx / 15.0 * 0.1

        confidence = min(1.0, adx_factor + abs(di_diff) / 50.0)

        # Momentum confirmation
        momentum = float(market_data.get('momentum', market_data.get('log_return', 0.0)))
        if np.sign(direction) == np.sign(momentum) and adx > 20:
            confidence = min(confidence * 1.2, 1.0)

        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                if adx > self.adx_threshold:
                    direction = direction * 0.6 + float(np.tanh(output[0, 1])) * 0.4
            except:
                pass
        
        self.record_instruments(
                adx=float(adx) if 'adx' in dir() else 0.0,
                plus_di=float(plus_di) if 'plus_di' in dir() else 0.0,
                minus_di=float(minus_di) if 'minus_di' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

