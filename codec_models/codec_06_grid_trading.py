"""
Codec 06: Grid Trading
Profits from oscillation within a defined price band.

Grid logic:
  - Compute a dynamic grid around the rolling VWAP / EMA midline
  - Measure where the current price sits within the grid band
  - Buy near lower grid levels, sell near upper grid levels
  - Widen grid in high-volatility regimes; narrow in low-vol
  - Use ATR-normalised band width to filter for oscillating markets
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


class Codec06(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'grid_trading'
        super().__init__(config)

        self.n_levels    = config.get('n_levels', 5)
        self.band_atr    = config.get('band_atr_mult', 2.0)
        self.window      = config.get('window', 20)

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            self.model = None

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price  = float(market_data.get('price', 1.0))
        high   = float(market_data.get('high', price))
        low    = float(market_data.get('low', price))
        atr    = float(market_data.get('atr_14', (high - low)))

        returns = features[:min(len(features), 64)]
        n = len(returns)

        if n < self.window:
            return self.validate_signal(0.15, 0.0)

        closes, highs, lows, volumes = self.get_ohlcv(market_data, features)


        prices = closes  # calibrated to pandas parquet data
        midline = float(np.mean(prices[-self.window:]))
        vol     = float(np.std(prices[-self.window:]))

        # Dynamic band width: ATR-scaled
        band_half = max(atr * self.band_atr, vol * 1.5, price * 0.005)

        lower = midline - band_half
        upper = midline + band_half

        if price <= lower or price >= upper:
            # Outside grid — no trade
            return self.validate_signal(0.1, 0.0)

        # Normalised position within band: -1 (at lower) to +1 (at upper)
        band_pos = 2.0 * (price - lower) / (upper - lower + 1e-8) - 1.0

        # Grid levels: evenly spaced; trade against band_pos
        level_step = 2.0 / self.n_levels
        level_idx  = int((band_pos + 1.0) / level_step)
        # Closer to lower level → buy; closer to upper → sell
        grid_signal = -band_pos   # mean reversion within band

        # Confidence higher when price is close to a grid edge
        edge_proximity = 1.0 - abs(band_pos % (level_step / 2.0))
        confidence = min(1.0, 0.3 + abs(grid_signal) * 0.5 + edge_proximity * 0.2)

        # Regime filter: only trade in oscillating (low Hurst) markets
        price_changes = np.diff(prices[-self.window:])
        sign_changes  = np.sum(np.diff(np.sign(price_changes)) != 0)
        oscillation   = sign_changes / max(len(price_changes) - 1, 1)
        confidence   *= (0.5 + oscillation)   # boost conviction when oscillating

        direction = float(np.tanh(grid_signal * 2.0))

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                direction  = direction * 0.6 + float(np.tanh(float(out[0, 0]))) * 0.4
                confidence = confidence * 0.6 + float(mx.sigmoid(out[0, 1])) * 0.4
            except Exception:
                pass

        self.record_instruments(
                band_pos=float(band_pos) if 'band_pos' in dir() else 0.0,
                oscillation=float(oscillation) if 'oscillation' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
