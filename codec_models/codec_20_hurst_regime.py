"""
Codec 20: Hurst Regime
Long-memory detection for trend vs mean-reversion regime.
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


class Codec20(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'hurst_regime'
        super().__init__(config)
        
        self.lookback = config.get('lookback', 100)
        self.hurst_memory = []
        
        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    
    def _estimate_hurst(self, returns: np.ndarray) -> float:
        if len(returns) < 20:
            return 0.5
        
        lags = range(2, min(len(returns) // 2, 50))
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
        
        if len(tau) < 2:
            return 0.5
        
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        
        slope = np.polyfit(log_lags, log_tau, 1)[0]
        return slope / 2
    
    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        returns = market_data.get('returns_history', [])
        price = market_data.get('price', 0)
        sma = market_data.get('sma_15', price)
        
        if isinstance(returns, np.ndarray) and len(returns) > 20:
            hurst = self._estimate_hurst(returns)
        else:
            hurst = 0.5
        
        direction = 0.0
        confidence = 0.2
        
        if hurst > 0.55:
            trend = price - sma if sma > 0 else 0
            direction = np.sign(trend) * min(abs(trend) / (price * 0.02), 1.0)
            confidence = min((hurst - 0.5) * 4 + 0.3, 1.0)
        elif hurst < 0.45:
            deviation = (price - sma) / sma if sma > 0 else 0
            direction = -np.sign(deviation) * min(abs(deviation) * 10, 1.0)
            confidence = min((0.5 - hurst) * 4 + 0.3, 1.0)
        else:
            confidence = 0.3
        
        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mx_features = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                output = self.model(mx_features)
                direction = direction * 0.5 + float(np.tanh(output[0, 1])) * 0.5
            except:
                pass
        
        self.record_instruments(
                hurst_exponent=float(hurst_exp) if 'hurst_exp' in dir() else 0.5,
            )
        return self.validate_signal(confidence, direction)
    
    def test_time_adapter(self, batch_data: Dict[str, Any], learning_rate: float = 1e-3) -> None:
        pass
    
    def online_adapter(self, *args, **kwargs) -> None:
        pass

