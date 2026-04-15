"""Core data models for the trading strategy system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Signal:
    pair: str
    signal_type: SignalType
    price: float
    confidence: float = 1.0
    reason: str = ""


@dataclass
class Ticker:
    pair: str
    price: float
    timestamp: Optional[float] = None
    volume: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()
