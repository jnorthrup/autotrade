"""Abstract interface for trading strategies."""

from abc import ABC, abstractmethod
from typing import List

from strategy.models import Ticker, Signal


class TradingStrategy(ABC):
    """Interface that all trading strategies must implement.

    A strategy consumes ticker data and emits buy/sell/hold signals.
    It maintains internal state (price history buffers) per trading pair
    and is completely decoupled from execution logic.
    """

    @abstractmethod
    def evaluate(self, tickers: List[Ticker]) -> List[Signal]:
        """Evaluate incoming ticker data and produce trading signals.

        Args:
            tickers: List of current ticker updates (one or more pairs).

        Returns:
            List of Signal objects, one per pair that was evaluated.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state (price history buffers)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this strategy."""
        ...
