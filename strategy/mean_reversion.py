"""Mean-Reversion strategy implementation."""

from collections import defaultdict
from typing import Dict, List

from strategy.interface import TradingStrategy
from strategy.models import Ticker, Signal, SignalType
from strategy.price_history import PriceHistoryBuffer


class MeanReversionStrategy(TradingStrategy):
    """Generates BUY/SELL/HOLD signals based on deviation from the mean price.

    When the current price is significantly below the lookback mean, the
    strategy emits BUY (expecting reversion upward). When the price is
    significantly above the mean, it emits SELL.

    Parameters:
        lookback:       Number of recent ticks used to compute the mean (>= 2).
        threshold_pct:  Percentage deviation from the mean required to trigger
                        a BUY or SELL signal (e.g. 0.02 = 2 %).
        price_buffer_size: Max ticks stored per pair (>= lookback).
    """

    def __init__(
        self,
        lookback: int = 20,
        threshold_pct: float = 0.02,
        price_buffer_size: int = 100,
    ):
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be > 0, got {threshold_pct}")
        if price_buffer_size < lookback:
            raise ValueError(
                f"price_buffer_size ({price_buffer_size}) must be >= lookback ({lookback})"
            )

        self._lookback = lookback
        self._threshold_pct = threshold_pct
        self._price_buffer_size = price_buffer_size
        self._histories: Dict[str, PriceHistoryBuffer] = {}

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return (
            f"MeanReversion(lookback={self._lookback}, "
            f"threshold={self._threshold_pct:.2%})"
        )

    @property
    def lookback(self) -> int:
        return self._lookback

    @property
    def threshold_pct(self) -> float:
        return self._threshold_pct

    # ------------------------------------------------------------------
    def _history(self, pair: str) -> PriceHistoryBuffer:
        if pair not in self._histories:
            self._histories[pair] = PriceHistoryBuffer(self._price_buffer_size)
        return self._histories[pair]

    def _compute_signal(self, pair: str, current_price: float) -> Signal:
        history = self._history(pair)
        history.append(current_price)

        if len(history) < self._lookback:
            return Signal(
                pair=pair,
                signal_type=SignalType.HOLD,
                price=current_price,
                confidence=0.0,
                reason=f"Insufficient data ({len(history)}/{self._lookback})",
            )

        mean_price = history.simple_moving_average(self._lookback)
        deviation = (current_price - mean_price) / mean_price  # positive = above mean

        if deviation > self._threshold_pct:
            return Signal(
                pair=pair,
                signal_type=SignalType.SELL,
                price=current_price,
                confidence=min(1.0, deviation / self._threshold_pct),
                reason=(
                    f"Price {current_price:.4f} is {deviation:.2%} above "
                    f"mean {mean_price:.4f} (threshold {self._threshold_pct:.2%})"
                ),
            )

        if deviation < -self._threshold_pct:
            return Signal(
                pair=pair,
                signal_type=SignalType.BUY,
                price=current_price,
                confidence=min(1.0, abs(deviation) / self._threshold_pct),
                reason=(
                    f"Price {current_price:.4f} is {abs(deviation):.2%} below "
                    f"mean {mean_price:.4f} (threshold {self._threshold_pct:.2%})"
                ),
            )

        return Signal(
            pair=pair,
            signal_type=SignalType.HOLD,
            price=current_price,
            confidence=0.5,
            reason=(
                f"Price {current_price:.4f} within {self._threshold_pct:.2%} "
                f"of mean {mean_price:.4f}"
            ),
        )

    # ------------------------------------------------------------------
    def evaluate(self, tickers: List[Ticker]) -> List[Signal]:
        return [self._compute_signal(t.pair, t.price) for t in tickers]

    def reset(self) -> None:
        self._histories.clear()
