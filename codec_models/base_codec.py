"""
BaseExpert — Abstract Base Class for all 24 SOTA Codec Experts
==============================================================

Defines the interface all expert codec implementations must follow.
Each expert must support:
1. forward     : generate trade signal from tick context + indicator vector
2. online_adapter : test-time adaptation (online fine-tuning via MLX)
3. update_ob_memory : fixed 512-bar temporal order book buffer management
4. MLX compatibility for Apple Silicon

Naming convention (crypto-technical):
    tick_context   : spot tick dict (price, volume, lob_delta, bid_ask_spread, etc.)
    indicator_vec  : computed technical indicator / feature vector [n_features]
    ob_memory      : 512-bar temporal order book buffer [512, 64] (ring buffer)
    trade_ledger   : running per-expert trade performance ledger
    signal_count   : total signals emitted by this expert
    cumulative_pnl : running sum of realized PnL across all signals
"""

from abc import ABC, abstractmethod
import os
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not available - using NumPy fallback")


class BaseExpert(ABC):
    """
    Abstract base class for all 24 SOTA codec expert implementations.

    Each expert must implement:
    - __init__         : initialise model and temporal order book buffer
    - forward          : generate trade signal (confidence, direction)
    - online_adapter   : test-time online fine-tuning via MLX value_and_grad
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config : expert-specific parameter dict
        """
        self.config = config
        self.name = config.get('name', 'base_expert')
        self.version = config.get('version', '1.0')

        # Temporal order book buffer: 512-bar ring buffer, 64-dim feature per bar
        self.ob_memory = np.zeros((512, 64), dtype=np.float32)
        self.ob_memory_idx = 0

        # ── Instrument predictor bus ──────────────────────────────────────
        # Each codec populates this dict in forward() with all raw indicator
        # values it computed (RSI, MACD, ATR, z-score, etc.).
        # train.py harvests these into the HRM feature matrix so the shared
        # TemporalOrderBook encoder can learn to predict indicator dynamics
        # (GOALS.md §3: "predicts next-bar codec features + all indicator kernels").
        self.instruments: Dict[str, float] = {}

        # Trade performance ledger
        self.trade_ledger = {
            'sharpe': 0.0,
            'hit_rate': 0.0,
            'cumulative_pnl': 0.0,
            'signal_count': 0,
        }

    @abstractmethod
    def forward(
        self,
        tick_context: Dict[str, Any],
        indicator_vec: np.ndarray
    ) -> Tuple[float, float]:
        """
        Generate trade signal from tick context and indicator vector.

        Args:
            tick_context : spot tick dict containing:
                - 'price'       : float — current mid price
                - 'volume'      : float — recent trading volume
                - 'lob_delta'   : float — LOB bid-ask quantity delta
                - 'bid_ask_spread' : float — bid-ask spread
                - 'funding_rate'   : float (optional) — perpetual funding rate
                - 'onchain_active_addresses' : float (optional) — on-chain activity
                - 'timestamp'   : datetime — bar timestamp
            indicator_vec : computed technical indicator vector [n_features]

        Returns:
            (conviction, direction) where:
                conviction : float ∈ [0, 1] — signal conviction
                direction  : float ∈ [-1, 1] — negative = short, positive = long
        """
        pass

    @abstractmethod
    def online_adapter(
        self,
        batch_data: Dict[str, Any],
        learning_rate: float = 1e-3
    ) -> None:
        """
        Online test-time adaptation via MLX value_and_grad.

        Args:
            batch_data    : dict with:
                - 'inputs'  : np.ndarray [batch, n_features]
                - 'targets' : np.ndarray [batch, 2] — [conviction, direction]
                - 'weights' : np.ndarray [batch] (optional — sample weights)
            learning_rate : low LR for stable online updates (1e-3 typical)
        """
        pass

    def update_ob_memory(self, direction: float, indicator_vec: np.ndarray) -> None:
        """
        Update the 512-bar temporal order book ring buffer.

        Args:
            direction     : trade direction signal
            indicator_vec : feature vector [64] for current bar
        """
        if len(indicator_vec) != 64:
            if len(indicator_vec) < 64:
                padded = np.zeros(64, dtype=np.float32)
                padded[:len(indicator_vec)] = indicator_vec
                indicator_vec = padded
            else:
                indicator_vec = indicator_vec[:64]

        self.ob_memory[self.ob_memory_idx] = indicator_vec
        self.ob_memory_idx = (self.ob_memory_idx + 1) % 512

    def reset_runtime_state(self) -> None:
        """
        Reset per-stream transient state when the input symbol/source changes.

        This avoids leaking one symbol's rolling context into the next symbol
        when train.py processes a concatenated multi-symbol DataFrame.
        """
        self.ob_memory.fill(0.0)
        self.ob_memory_idx = 0
        self.instruments.clear()

    def get_ob_summary(self) -> Dict[str, float]:
        """
        Summary statistics of the temporal order book buffer.

        Returns:
            Dict with mean, std, q25, q75, bar_count
        """
        if self.ob_memory_idx == 0:
            return {}

        buf = self.ob_memory[:self.ob_memory_idx]
        return {
            'mean': float(np.mean(buf)),
            'std': float(np.std(buf)),
            'q25': float(np.quantile(buf, 0.25)),
            'q75': float(np.quantile(buf, 0.75)),
            'bar_count': self.ob_memory_idx,
        }

    def record_trade_outcome(
        self,
        realized_pnl: float,
        direction: float,
        actual_return: float
    ) -> None:
        """
        Record a completed trade outcome into the trade ledger.

        Args:
            realized_pnl  : PnL for this signal
            direction     : predicted direction
            actual_return : actual market return
        """
        self.trade_ledger['cumulative_pnl'] += realized_pnl
        self.trade_ledger['signal_count'] += 1

        n = self.trade_ledger['signal_count']
        if realized_pnl > 0:
            self.trade_ledger['hit_rate'] = (
                (self.trade_ledger['hit_rate'] * (n - 1) + 1) / n
            )
        else:
            self.trade_ledger['hit_rate'] = (
                (self.trade_ledger['hit_rate'] * (n - 1)) / n
            )

    def get_trade_ledger(self) -> Dict[str, Any]:
        """Return current trade ledger with Sharpe estimate."""
        ledger = self.trade_ledger.copy()
        if ledger['signal_count'] >= 10:
            ledger['sharpe'] = ledger['cumulative_pnl'] / max(1.0, ledger['signal_count'])
        return ledger

    def reset_trade_ledger(self) -> None:
        """Reset the trade ledger to zero."""
        self.trade_ledger = {
            'sharpe': 0.0,
            'hit_rate': 0.0,
            'cumulative_pnl': 0.0,
            'signal_count': 0,
        }

    def get_ohlcv(self, market_data: dict, features: np.ndarray):
        """
        Return real rolling OHLCV arrays from market_data when available
        (populated by train.py's compute_signals from the actual pandas parquet data),
        falling back to reconstruction from the return series in `features`.

        Returns:
            closes, highs, lows, volumes  — each a np.ndarray of shape [n_bars]
        """
        closes  = market_data.get('closes')
        highs   = market_data.get('highs')
        lows    = market_data.get('lows')
        volumes = market_data.get('volumes')

        if closes is not None and len(closes) > 1:
            return (
                np.asarray(closes,  dtype=np.float32),
                np.asarray(highs,   dtype=np.float32),
                np.asarray(lows,    dtype=np.float32),
                np.asarray(volumes, dtype=np.float32),
            )

        # Fallback: reconstruct price series from returns in features
        price = float(market_data.get('price', 1.0))
        returns = features[:min(len(features), 64)]
        if len(returns) > 0:
            reconstructed = price * np.exp(np.cumsum(-returns[::-1])[::-1])
        else:
            reconstructed = np.array([price], dtype=np.float32)
        n = len(reconstructed)
        h  = float(market_data.get('high', price))
        l  = float(market_data.get('low',  price))
        return (
            reconstructed.astype(np.float32),
            np.full(n, h, dtype=np.float32),
            np.full(n, l, dtype=np.float32),
            np.ones(n, dtype=np.float32),
        )

    def record_instruments(self, **kwargs: float) -> None:

        """
        Populate self.instruments with named indicator readings.

        Call this inside forward() before returning:
            self.record_instruments(rsi_14=rsi14, macd_hist=hist, atr_norm=atr)

        train.py collects these values into the HRM feature matrix so the
        shared encoder can learn to predict each indicator's next value.
        """
        self.instruments.update({k: float(v) for k, v in kwargs.items()})

    def validate_signal(self, conviction: float, direction: float) -> Tuple[float, float]:
        """Clip and validate signal output to valid ranges."""
        conviction = max(0.0, min(1.0, float(conviction)))
        direction = max(-1.0, min(1.0, float(direction)))
        return conviction, direction

    def __repr__(self) -> str:
        return f"{self.name} (v{self.version})"


# Legacy alias — BaseCodec was the old name
BaseCodec = BaseExpert


class ExpertFactory:
    """Factory for creating codec expert instances by ID."""

    @staticmethod
    def create_expert(expert_id: int, config: Dict[str, Any] = None) -> BaseExpert:
        """
        Create a codec expert by ID (1–24).

        Args:
            expert_id : expert number (1–24)
            config    : expert configuration dict

        Returns:
            Codec expert instance
        """
        if config is None:
            config = {}

        config['codec_id'] = expert_id
        config['name'] = f"codec_{expert_id:02d}"

        # Try to import the specific codec module from codec_models
        import importlib
        import glob
        codec_dir = os.path.dirname(__file__) if '__file__' in dir() else os.path.dirname(os.path.abspath(__file__))
        
        # Find the codec module file matching this ID
        codec_files = sorted(glob.glob(os.path.join(codec_dir, f"codec_{expert_id:02d}_*.py")))
        
        for file_path in codec_files:
            module_name = os.path.basename(file_path)[:-3]
            try:
                module = importlib.import_module(f".{module_name}", package="codec_models")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, BaseExpert) and attr is not BaseExpert:
                        return attr(config)
            except (ImportError, AttributeError):
                continue

        # Fallback to generic codec
        from .codec_generic import GenericCodec
        return GenericCodec(config)


# Legacy alias
CodecFactory = ExpertFactory


def get_expert_panel(config: Dict[str, Any] = None) -> list:
    """
    Instantiate all 24 codec experts.

    Returns:
        List of 24 BaseExpert instances (full codec panel)
    """
    if config is None:
        config = {}
    return [ExpertFactory.create_expert(i, config) for i in range(1, 25)]


# Legacy alias
get_all_codecs = get_expert_panel
