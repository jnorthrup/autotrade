"""Moving-Average Crossover strategy implementation."""

from collections import defaultdict
from typing import Dict, List

from strategy.interface import TradingStrategy
from strategy.models import Ticker, Signal, SignalType
from strategy.price_history import PriceHistoryBuffer


class MovingAverageCrossoverStrategy(TradingStrategy):
    """Generates BUY/SELL/HOLD signals based on short-term vs long-term SMA crossover.

    Parameters:
        short_window:  Look-back period for the short-term SMA (must be >= 2).
        long_window:   Look-back period for the long-term SMA (must be > short_window).
        price_buffer_size: Max ticks stored per pair (should be >= long_window).
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        price_buffer_size: int = 100,
    ):
        if short_window < 2:
            raise ValueError(f"short_window must be >= 2, got {short_window}")
        if long_window <= short_window:
            raise ValueError(
                f"long_window ({long_window}) must be > short_window ({short_window})"
            )
        if price_buffer_size < long_window:
            raise ValueError(
                f"price_buffer_size ({price_buffer_size}) must be >= long_window ({long_window})"
            )

        self._short_window = short_window
        self._long_window = long_window
        self._price_buffer_size = price_buffer_size

        self._histories: Dict[str, PriceHistoryBuffer] = {}
        self._prev_short_above_long: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return (
            f"MovingAverageCrossover(short={self._short_window}, "
            f"long={self._long_window})"
        )

    @property
    def short_window(self) -> int:
        return self._short_window

    @property
    def long_window(self) -> int:
        return self._long_window

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _history(self, pair: str) -> PriceHistoryBuffer:
        if pair not in self._histories:
            self._histories[pair] = PriceHistoryBuffer(self._price_buffer_size)
        return self._histories[pair]

    def _compute_signals(self, pair: str, current_price: float) -> Signal:
        history = self._history(pair)
        history.append(current_price)

        # Not enough data yet for the long window — hold.
        if len(history) < self._long_window:
            return Signal(
                pair=pair,
                signal_type=SignalType.HOLD,
                price=current_price,
                confidence=0.0,
                reason=f"Insufficient data ({len(history)}/{self._long_window})",
            )

        short_ma = history.simple_moving_average(self._short_window)
        long_ma = history.simple_moving_average(self._long_window)

        short_above_long = short_ma > long_ma

        # Determine crossover direction
        prev = self._prev_short_above_long.get(pair)
        self._prev_short_above_long[pair] = short_above_long

        if prev is None:
            # First time we have enough data — no crossover to detect yet.
            return Signal(
                pair=pair,
                signal_type=SignalType.HOLD,
                price=current_price,
                confidence=0.5,
                reason="Baseline established, waiting for crossover",
            )

        if short_above_long and not prev:
            # Short crossed ABOVE long → bullish crossover → BUY
            return Signal(
                pair=pair,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=min(1.0, abs(short_ma - long_ma) / long_ma * 100),
                reason=f"Bullish crossover: short_ma={short_ma:.4f} > long_ma={long_ma:.4f}",
            )

        if not short_above_long and prev:
            # Short crossed BELOW long → bearish crossover → SELL
            return Signal(
                pair=pair,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=min(1.0, abs(short_ma - long_ma) / long_ma * 100),
                reason=f"Bearish crossover: short_ma={short_ma:.4f} < long_ma={long_ma:.4f}",
            )

        # No crossover — hold
        return Signal(
            pair=pair,
            signal_type=SignalType.HOLD,
            price=current_price,
            confidence=0.5,
            reason="No crossover detected",
        )

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------
    def evaluate(self, tickers: List[Ticker]) -> List[Signal]:
        return [self._compute_signals(t.pair, t.price) for t in tickers]

    def reset(self) -> None:
        self._histories.clear()
        self._prev_short_above_long.clear()
