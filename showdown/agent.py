"""
showdown.agent – Agent adapter wrapping any BaseExpert codec
=============================================================

Each Agent owns:
  - Its own IndicatorComputer (isolated rolling indicator state)
  - A virtual portfolio (cash + holdings per pair)
  - A trade history ledger

The adapter translates (conviction, direction) from codec.forward()
into discrete BUY / SELL / HOLD decisions with position sizing.

It also populates the 64-element indicator_vec that codecs expect,
derived from the indicator layer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .indicators import IndicatorComputer
from codec_models.base_codec import BaseExpert


# ── Action constants ────────────────────────────────────────────────────

ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"


# ── Indicator-vector builder ────────────────────────────────────────────

def build_indicator_vec(market_data: Dict[str, Any]) -> np.ndarray:
    """
    Construct a 64-element indicator feature vector from the market_data
    dict produced by IndicatorComputer.

    Layout (64 slots):
        [0]    price / sma_20
        [1]    price / sma_15
        [2]    price / ema_12
        [3]    price / ema_26
        [4]    macd / price  (scaled)
        [5]    macd_signal / price
        [6]    macd_hist / price
        [7]    rsi / 100
        [8]    (price - bb_lower) / (bb_upper - bb_lower)  or 0.5
        [9]    (bb_upper - bb_lower) / bb_mid              or 0
        [10]   atr_14 / price
        [11]   stoch_k / 100
        [12]   stoch_d / 100
        [13]   adx / 100
        [14]   plus_di / 100
        [15]   minus_di / 100
        [16]   price / vwap  (or 1.0)
        [17]   momentum / 100  (scaled)
        [18]   volume / avg_volume  (or 1.0)
        [19]   log_return
        [20-39]  zero-padded (reserved for higher-resolution features)
        [40-63]  zero-padded (reserved for order-book / on-chain features)
    """
    vec = np.zeros(64, dtype=np.float32)

    price = float(market_data.get("price", 1.0))
    safe_price = price if price != 0.0 else 1.0

    vec[0] = price / float(market_data.get("sma_20", price) or price or 1.0)
    vec[1] = price / float(market_data.get("sma_15", price) or price or 1.0)
    vec[2] = price / float(market_data.get("ema_12", price) or price or 1.0)
    vec[3] = price / float(market_data.get("ema_26", price) or price or 1.0)

    vec[4] = float(market_data.get("macd", 0.0)) / safe_price
    vec[5] = float(market_data.get("macd_signal", 0.0)) / safe_price
    vec[6] = float(market_data.get("macd_hist", 0.0)) / safe_price

    vec[7] = float(market_data.get("rsi", 50.0)) / 100.0

    bb_upper = float(market_data.get("bb_upper", price))
    bb_lower = float(market_data.get("bb_lower", price))
    bb_mid = float(market_data.get("bb_mid", price))
    bb_width = bb_upper - bb_lower
    if bb_width > 0:
        vec[8] = (price - bb_lower) / bb_width
    else:
        vec[8] = 0.5
    vec[9] = bb_width / bb_mid if bb_mid != 0.0 else 0.0

    vec[10] = float(market_data.get("atr_14", 0.0)) / safe_price
    vec[11] = float(market_data.get("stoch_k", 50.0)) / 100.0
    vec[12] = float(market_data.get("stoch_d", 50.0)) / 100.0
    vec[13] = float(market_data.get("adx", 0.0)) / 100.0
    vec[14] = float(market_data.get("plus_di", 0.0)) / 100.0
    vec[15] = float(market_data.get("minus_di", 0.0)) / 100.0

    vwap = float(market_data.get("vwap", price))
    vec[16] = price / vwap if vwap != 0.0 else 1.0

    vec[17] = float(market_data.get("momentum", 0.0)) / 100.0

    avg_vol = float(market_data.get("avg_volume", 0.0))
    vol = float(market_data.get("volume", 0.0))
    vec[18] = vol / avg_vol if avg_vol > 0.0 else 1.0

    vec[19] = float(market_data.get("log_return", 0.0))

    # Slots 20-63 remain zero (reserved for future use)
    return vec


# ── TradeAction dataclass-like dict ─────────────────────────────────────

def make_trade_action(
    pair: str,
    action: str,
    size: float = 0.0,
    price: float = 0.0,
    conviction: float = 0.0,
    direction: float = 0.0,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return a standardised trade-action dict.

    Keys: pair, action, size, price, conviction, direction, timestamp
    """
    return {
        "pair": pair,
        "action": action,
        "size": size,
        "price": price,
        "conviction": conviction,
        "direction": direction,
        "timestamp": timestamp or time.time(),
    }


# ── Agent ───────────────────────────────────────────────────────────────

class Agent:
    """
    Standardised agent that wraps a BaseExpert codec for the showdown
    harness.

    Each agent maintains **isolated** state:
      - Its own ``IndicatorComputer`` (rolling indicator buffers per pair)
      - A virtual portfolio: ``cash`` + ``holdings`` dict (pair -> qty)
      - A ``trade_history`` list of action dicts

    Parameters
    ----------
    codec : BaseExpert
        Any codec expert implementing ``forward(market_data, indicator_vec)``.
    initial_cash : float
        Starting virtual cash balance (default 100 000).
    conviction_threshold : float
        Minimum conviction to act on a signal (default 0.4).
    position_fraction : float
        Fraction of available cash to use per BUY (default 0.25).
    """

    def __init__(
        self,
        codec: BaseExpert,
        initial_cash: float = 100_000.0,
        conviction_threshold: float = 0.4,
        position_fraction: float = 0.25,
    ) -> None:
        self.codec = codec
        self.initial_cash = initial_cash
        self.conviction_threshold = conviction_threshold
        self.position_fraction = position_fraction

        # Isolated indicator computer (one per agent so buffers don't leak)
        self._indicator_computer = IndicatorComputer(buffer_size=200)

        # Virtual portfolio
        self.cash: float = initial_cash
        self.holdings: Dict[str, float] = {}  # pair -> quantity held

        # Trade history (list of action dicts)
        self.trade_history: List[Dict[str, Any]] = []

    # ── Public API ──────────────────────────────────────────────────────

    def on_tick(self, tickers: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process one batch of tick data and return a list of trade actions.

        Parameters
        ----------
        tickers : dict
            Mapping of ``pair`` -> tick dict.  Each tick dict must contain
            at minimum ``'price'`` (float) and may contain ``'volume'`` (float).

        Returns
        -------
        list[dict]
            One trade-action dict per pair (BUY / SELL / HOLD).
        """
        actions: List[Dict[str, Any]] = []

        for pair, tick in tickers.items():
            price = float(tick.get("price", 0.0))
            volume = float(tick.get("volume", 0.0))

            # 1. Compute indicators via the isolated IndicatorComputer
            market_data = self._indicator_computer.compute(pair, price, volume)

            # 2. Build the 64-element indicator_vec
            indicator_vec = build_indicator_vec(market_data)

            # 3. Feed to codec forward
            conviction, direction = self.codec.forward(market_data, indicator_vec)

            # Also update codec's OB memory for temporal context
            self.codec.update_ob_memory(direction, indicator_vec)

            # 4. Translate (conviction, direction) -> BUY / SELL / HOLD
            action_str = ACTION_HOLD
            size = 0.0

            if conviction > self.conviction_threshold:
                if direction > 0:
                    # BUY signal
                    action_str = ACTION_BUY
                    spend = self.cash * self.position_fraction * conviction
                    if spend > 0 and price > 0:
                        size = spend / price
                        self.cash -= size * price
                        self.holdings[pair] = self.holdings.get(pair, 0.0) + size
                elif direction < 0:
                    # SELL signal
                    action_str = ACTION_SELL
                    held = self.holdings.get(pair, 0.0)
                    if held > 0 and price > 0:
                        sell_frac = conviction * self.position_fraction
                        size = held * sell_frac
                        if size > 0:
                            self.cash += size * price
                            self.holdings[pair] = held - size
                            if self.holdings[pair] < 1e-12:
                                self.holdings[pair] = 0.0

            action = make_trade_action(
                pair=pair,
                action=action_str,
                size=size,
                price=price,
                conviction=conviction,
                direction=direction,
            )
            actions.append(action)
            self.trade_history.append(action)

        return actions

    def get_portfolio(self) -> Dict[str, Any]:
        """
        Return a snapshot of the virtual portfolio.

        Keys: cash, holdings, trade_history, total_trades, codec_name
        """
        return {
            "cash": self.cash,
            "holdings": dict(self.holdings),
            "trade_history": list(self.trade_history),
            "total_trades": len(self.trade_history),
            "codec_name": self.codec.name,
        }

    def reset(self) -> None:
        """Reset agent to initial state (fresh portfolio, empty history)."""
        self.cash = self.initial_cash
        self.holdings.clear()
        self.trade_history.clear()
        self._indicator_computer = IndicatorComputer(buffer_size=200)
        self.codec.reset_runtime_state()
        self.codec.reset_trade_ledger()

    # ── Helpers ─────────────────────────────────────────────────────────

    def portfolio_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """
        Compute total portfolio value (cash + market value of holdings).

        Parameters
        ----------
        prices : dict, optional
            Mapping of pair -> current price.  If not provided, uses
            the last known price from holdings (assumes zero).
        """
        total = self.cash
        if prices:
            for pair, qty in self.holdings.items():
                total += qty * prices.get(pair, 0.0)
        return total

    def __repr__(self) -> str:
        return (
            f"Agent(codec={self.codec.name}, cash={self.cash:.2f}, "
            f"positions={len(self.holdings)})"
        )
