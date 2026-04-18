"""
Codec 07: Volume Profile
Volume-weighted price distribution and VPOC (Volume Point of Control) analysis.

Signal logic:
  - Reconstruct an approximate price-volume distribution from the OHLCV stream
  - Identify the VPOC (price level with highest volume density)
  - Compute the current price's distance from VPOC (in z-score units)
  - Price below VPOC → potential buy; above VPOC → potential sell
  - Conviction proportional to |distance| and cumulative volume skew
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


class Codec07(BaseCodec):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['name'] = 'volume_profile'
        super().__init__(config)

        self.n_bins  = config.get('n_bins', 20)
        self.window  = config.get('window', 50)

        if HAS_MLX:
            self.model = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            self.model = None

    def forward(self, market_data: Dict[str, Any], features: np.ndarray) -> Tuple[float, float]:
        price   = float(market_data.get('price', 1.0))
        high    = float(market_data.get('high', price))
        low     = float(market_data.get('low', price))
        volume  = float(market_data.get('volume', 1.0))

        returns = features[:min(len(features), 64)]
        n = len(returns)

        if n < 10:
            return self.validate_signal(0.15, 0.0)

        closes, highs, lows, volumes = self.get_ohlcv(market_data, features)


        prices = closes  # calibrated to pandas parquet data
        w       = min(n, self.window)
        p_seg   = prices[-w:]

        # Approximate volume weights: use squared return as proxy
        #   (large moves = high volume participation)
        ret_seg   = returns[-w:]
        vol_proxy = np.abs(ret_seg) + 1e-6
        vol_proxy /= vol_proxy.sum()

        # Volume-weighted histogram → VPOC
        price_min = p_seg.min()
        price_max = p_seg.max() + 1e-8
        bins = np.linspace(price_min, price_max, self.n_bins + 1)
        vol_hist = np.zeros(self.n_bins)
        if price_max <= price_min + 1e-8:
            return self.validate_signal(0.15, 0.0)

        for i, p in enumerate(p_seg):
            b = min(int((p - price_min) / (price_max - price_min) * self.n_bins),
                    self.n_bins - 1)
            vol_hist[b] += vol_proxy[i]

        vpoc_bin  = int(np.argmax(vol_hist))
        vpoc_price = float(bins[vpoc_bin] + (bins[vpoc_bin + 1] - bins[vpoc_bin]) / 2.0)

        # Distance from VPOC in normalised units
        price_std = float(p_seg.std()) + 1e-8
        dist_z    = (price - vpoc_price) / price_std

        # Mean-reversion toward VPOC
        direction  = -float(np.sign(dist_z)) if abs(dist_z) > 0.3 else 0.0

        # Volume skew: is more volume above or below current price?
        above_bin = min(int((price - price_min) / (price_max - price_min) * self.n_bins),
                        self.n_bins - 1)
        vol_above = vol_hist[above_bin + 1:].sum()
        vol_below = vol_hist[:above_bin].sum()
        vol_skew  = float(vol_below - vol_above)   # positive → more vol below → buy pressure

        confidence = min(1.0, (abs(dist_z) * 0.4 + abs(vol_skew) * 0.4 + 0.2))

        if HAS_MLX and self.model is not None and len(features) >= 64:
            try:
                mlx_in = mx.array(features[:64].reshape(1, -1).astype(np.float32))
                out = self.model(mlx_in)
                direction  = direction * 0.6 + float(np.tanh(float(out[0, 0]))) * 0.4
                confidence = confidence * 0.6 + float(mx.sigmoid(out[0, 1])) * 0.4
            except Exception:
                pass

        self.record_instruments(
                vpoc_dist_z=float(dist_z) if 'dist_z' in dir() else 0.0,
                vol_skew=float(vol_skew) if 'vol_skew' in dir() else 0.0,
            )
        return self.validate_signal(confidence, direction)

    def online_adapter(self, *args, **kwargs) -> None:
        pass
