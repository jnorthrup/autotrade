"""Circular price-history buffer maintained per trading pair."""

from collections import deque
from typing import Optional


class PriceHistoryBuffer:
    """Maintains a bounded rolling window of recent prices for a single pair."""

    def __init__(self, maxlen: int):
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._maxlen = maxlen
        self._prices: deque = deque(maxlen=maxlen)

    def append(self, price: float) -> None:
        self._prices.append(price)

    @property
    def prices(self) -> list:
        return list(self._prices)

    def __len__(self) -> int:
        return len(self._prices)

    @property
    def is_full(self) -> bool:
        return len(self._prices) == self._maxlen

    def simple_moving_average(self, period: Optional[int] = None) -> float:
        """Return the SMA over the last `period` prices (or all if None)."""
        if not self._prices:
            raise ValueError("No prices available")
        subset = self._prices if period is None else list(self._prices)[-period:]
        if not subset:
            raise ValueError(f"Requested period {period} exceeds available data")
        return sum(subset) / len(subset)

    def clear(self) -> None:
        self._prices.clear()
