"""
showdown.indicators – Rolling technical-indicator computer
==========================================================

Maintains a per-pair rolling OHLCV buffer and exposes
``compute_market_data(pair, price, volume) -> dict`` which returns
a fully-populated market_data dict for downstream codec agents.

All indicators are computed with pure NumPy (no TA-Lib dependency).

Implemented indicators
----------------------
SMA(15), SMA(20)
EMA(12), EMA(26)
MACD line / signal / histogram   (12/26/9)
RSI(14)
Bollinger Bands(20, 2)
ATR(14)
Stochastic %K(14) / %D(3)
ADX(14) with +DI / -DI
VWAP (rolling cumulative)
Momentum (10-bar rate of change)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np


# ── tiny helpers ────────────────────────────────────────────────────────

def _sma(arr: np.ndarray, period: int) -> float:
    """Simple moving average of the last *period* values in *arr*."""
    if len(arr) < period:
        return float(arr[-1]) if len(arr) else 0.0
    return float(np.mean(arr[-period:]))


def _ema(arr: np.ndarray, period: int) -> float:
    """Exponential moving average over the full array, returning the last value."""
    n = len(arr)
    if n == 0:
        return 0.0
    if n == 1:
        return float(arr[0])
    k = 2.0 / (period + 1)
    ema_val = float(arr[0])
    for i in range(1, n):
        ema_val = arr[i] * k + ema_val * (1.0 - k)
    return ema_val


def _rolling_ema_series(arr: np.ndarray, period: int) -> np.ndarray:
    """Return the full EMA series for *arr*."""
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """True-range array (length = len(highs)-1)."""
    if len(highs) < 2:
        return np.array([], dtype=np.float64)
    prev_close = closes[:-1]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - prev_close),
            np.abs(lows[1:] - prev_close),
        ),
    )
    return tr


# ── Per-pair bar buffer ───────────────────────────────────────────────

class _BarBuffer:
    """Fixed-length circular buffer holding OHLCV bars for one pair."""

    __slots__ = (
        "maxlen",
        "opens",
        "highs",
        "lows",
        "closes",
        "volumes",
        "typical_prices",
        "_current_open",
        "_current_high",
        "_current_low",
        "_current_volume",
        "_bar_started",
    )

    def __init__(self, maxlen: int = 200) -> None:
        self.maxlen = maxlen
        # main arrays (deques would work but numpy is faster for computation)
        self.opens: List[float] = []
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.volumes: List[float] = []
        self.typical_prices: List[float] = []
        # intrabar accumulation
        self._current_open: float = 0.0
        self._current_high: float = 0.0
        self._current_low: float = float("inf")
        self._current_volume: float = 0.0
        self._bar_started: bool = False

    # -- tick-level update (each call = one tick; we treat every tick as a bar) --
    def append_tick(self, price: float, volume: float) -> None:
        """Add a new bar (tick). Each tick is treated as one complete bar."""
        self.opens.append(price)
        self.highs.append(price)
        self.lows.append(price)
        self.closes.append(price)
        self.volumes.append(volume)
        self.typical_prices.append(price)  # H+L+C/3 = price when H=L=C=price
        # trim to maxlen
        if len(self.closes) > self.maxlen:
            excess = len(self.closes) - self.maxlen
            del self.opens[:excess]
            del self.highs[:excess]
            del self.lows[:excess]
            del self.closes[:excess]
            del self.volumes[:excess]
            del self.typical_prices[:excess]

    def as_arrays(self):
        return (
            np.asarray(self.opens, dtype=np.float64),
            np.asarray(self.highs, dtype=np.float64),
            np.asarray(self.lows, dtype=np.float64),
            np.asarray(self.closes, dtype=np.float64),
            np.asarray(self.volumes, dtype=np.float64),
            np.asarray(self.typical_prices, dtype=np.float64),
        )


# ── Main indicator computer ───────────────────────────────────────────

class IndicatorComputer:
    """
    Thread-safe indicator computer that maintains per-pair rolling
    OHLCV buffers and computes all required indicators on each tick.
    """

    def __init__(self, buffer_size: int = 200) -> None:
        self.buffer_size = buffer_size
        self._buffers: Dict[str, _BarBuffer] = {}
        self._lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def compute(self, pair: str, price: float, volume: float) -> Dict[str, Any]:
        """
        Record a new tick and return a market_data dict with all indicators.
        """
        with self._lock:
            if pair not in self._buffers:
                self._buffers[pair] = _BarBuffer(maxlen=self.buffer_size)
            buf = self._buffers[pair]
            buf.append_tick(price, volume)
            return self._compute_indicators(buf, price, volume)

    # -- internal computation ----------------------------------------------

    @staticmethod
    def _compute_indicators(buf: _BarBuffer, price: float, volume: float) -> Dict[str, Any]:
        opens, highs, lows, closes, volumes, typical = buf.as_arrays()
        n = len(closes)

        result: Dict[str, Any] = {
            "price": float(price),
            "high": float(price),
            "low": float(price),
            "open": float(price) if n > 0 else float(price),
            "volume": float(volume),
            "pair": "",
        }

        if n > 0:
            result["open"] = float(opens[-1])
            result["high"] = float(highs[-1])
            result["low"] = float(lows[-1])

        # ── SMA ──────────────────────────────────────────────────────
        result["sma_15"] = _sma(closes, 15)
        result["sma_20"] = _sma(closes, 20)

        # ── EMA ──────────────────────────────────────────────────────
        result["ema_12"] = _ema(closes, 12)
        result["ema_26"] = _ema(closes, 26)

        # ── MACD (12, 26, 9) ────────────────────────────────────────
        macd_line = result["ema_12"] - result["ema_26"]
        # Compute a proper signal line: EMA-9 of MACD series
        if n >= 2:
            ema12_series = _rolling_ema_series(closes, 12)
            ema26_series = _rolling_ema_series(closes, 26)
            macd_series = ema12_series - ema26_series
            signal_series = _rolling_ema_series(macd_series, 9)
            macd_signal = float(signal_series[-1])
        else:
            macd_signal = 0.0
        macd_hist = macd_line - macd_signal

        result["macd"] = float(macd_line)
        result["macd_signal"] = float(macd_signal)
        result["macd_hist"] = float(macd_hist)

        # ── RSI(14) ─────────────────────────────────────────────────
        result["rsi"] = IndicatorComputer._rsi(closes, 14)
        result["rsi_14"] = result["rsi"]

        # ── Bollinger Bands (20, 2) ─────────────────────────────────
        if n >= 20:
            window = closes[-20:]
            bb_mid = float(np.mean(window))
            bb_std = float(np.std(window, ddof=0))
            result["bb_mid"] = bb_mid
            result["bb_upper"] = bb_mid + 2.0 * bb_std
            result["bb_lower"] = bb_mid - 2.0 * bb_std
        else:
            bb_mid = _sma(closes, n) if n > 0 else price
            result["bb_mid"] = bb_mid
            result["bb_upper"] = bb_mid
            result["bb_lower"] = bb_mid

        # ── ATR(14) ─────────────────────────────────────────────────
        result["atr_14"] = IndicatorComputer._atr(highs, lows, closes, 14)

        # ── Stochastic %K(14) / %D(3) ───────────────────────────────
        stoch_k, stoch_d = IndicatorComputer._stochastic(highs, lows, closes, 14, 3)
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d

        # ── ADX(14) with +DI / -DI ──────────────────────────────────
        adx_val, plus_di, minus_di = IndicatorComputer._adx(highs, lows, closes, 14)
        result["adx"] = adx_val
        result["adx_14"] = adx_val
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di

        # ── VWAP (rolling cumulative) ────────────────────────────────
        result["vwap"] = IndicatorComputer._vwap(typical, volumes)

        # ── Momentum (10-bar ROC) ───────────────────────────────────
        result["momentum"] = IndicatorComputer._momentum(closes, 10)

        # ── Average volume (20-bar) ─────────────────────────────────
        result["avg_volume"] = _sma(volumes, 20)

        # ── Log return ──────────────────────────────────────────────
        if n >= 2:
            result["log_return"] = float(np.log(closes[-1] / closes[-2]))
        else:
            result["log_return"] = 0.0

        return result

    # ── Static indicator functions ────────────────────────────────────

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        """Wilder-smoothed RSI."""
        n = len(closes)
        if n < period + 1:
            # Not enough data: return neutral
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Initial average
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        # Wilder smoothing
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
            avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period

        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int = 14) -> float:
        """Average True Range (Wilder-smoothed)."""
        tr = _true_range(highs, lows, closes)
        if len(tr) == 0:
            return 0.0
        if len(tr) < period:
            return float(np.mean(tr))
        # Initial ATR
        atr_val = float(np.mean(tr[:period]))
        for i in range(period, len(tr)):
            atr_val = (atr_val * (period - 1) + float(tr[i])) / period
        return atr_val

    @staticmethod
    def _stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    k_period: int = 14, d_period: int = 3):
        """Stochastic %K and %D."""
        n = len(closes)
        if n < k_period:
            return 50.0, 50.0

        k_values = []
        for i in range(k_period - 1, n):
            window_high = float(np.max(highs[i - k_period + 1: i + 1]))
            window_low = float(np.min(lows[i - k_period + 1: i + 1]))
            denom = window_high - window_low
            if denom == 0.0:
                k_values.append(50.0)
            else:
                k_values.append((closes[i] - window_low) / denom * 100.0)

        k_now = k_values[-1]
        # %D is SMA of %K over d_period
        d_values = k_values[-d_period:]
        d_now = float(np.mean(d_values))
        return float(k_now), float(d_now)

    @staticmethod
    def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int = 14):
        """ADX with +DI and -DI (Wilder smoothing)."""
        n = len(closes)
        if n < period + 1:
            return 0.0, 0.0, 0.0

        # True range
        tr = _true_range(highs, lows, closes)

        # Directional movement
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth with Wilder method
        def _wilder_smooth(arr, p):
            result = []
            s = float(np.sum(arr[:p]))
            result.append(s)
            for i in range(p, len(arr)):
                s = s - s / p + float(arr[i])
                result.append(s)
            return np.array(result, dtype=np.float64)

        if len(tr) < period:
            return 0.0, 0.0, 0.0

        smooth_tr = _wilder_smooth(tr, period)
        smooth_plus_dm = _wilder_smooth(plus_dm, period)
        smooth_minus_dm = _wilder_smooth(minus_dm, period)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di_raw = np.where(smooth_tr > 0, smooth_plus_dm / smooth_tr * 100.0, 0.0)
            minus_di_raw = np.where(smooth_tr > 0, smooth_minus_dm / smooth_tr * 100.0, 0.0)
            dx = np.where(
                (plus_di_raw + minus_di_raw) > 0,
                np.abs(plus_di_raw - minus_di_raw) / (plus_di_raw + minus_di_raw) * 100.0,
                0.0,
            )

        # ADX = smoothed DX
        if len(dx) < period:
            adx_val = float(np.mean(dx)) if len(dx) > 0 else 0.0
        else:
            adx_val = float(np.mean(dx[:period]))
            for i in range(period, len(dx)):
                adx_val = (adx_val * (period - 1) + float(dx[i])) / period

        return (
            float(adx_val),
            float(plus_di_raw[-1]) if len(plus_di_raw) > 0 else 0.0,
            float(minus_di_raw[-1]) if len(minus_di_raw) > 0 else 0.0,
        )

    @staticmethod
    def _vwap(typical: np.ndarray, volumes: np.ndarray) -> float:
        """Cumulative VWAP over the buffer."""
        if len(volumes) == 0 or np.sum(volumes) == 0:
            return float(typical[-1]) if len(typical) > 0 else 0.0
        cum_tp_vol = np.cumsum(typical * volumes)
        cum_vol = np.cumsum(volumes)
        return float(cum_tp_vol[-1] / cum_vol[-1])

    @staticmethod
    def _momentum(closes: np.ndarray, period: int = 10) -> float:
        """Rate-of-change momentum (price / price_n - 1) * 100."""
        n = len(closes)
        if n <= period or closes[-1 - period] == 0:
            return 0.0
        return float((closes[-1] / closes[-1 - period] - 1.0) * 100.0)


# ── Module-level convenience singleton ────────────────────────────────

_default_computer = IndicatorComputer(buffer_size=200)
_default_lock = threading.Lock()


def compute_market_data(pair: str, price: float, volume: float) -> Dict[str, Any]:
    """
    Module-level convenience function.

    Maintains a global rolling buffer per *pair* (default 200 bars)
    and returns a fully populated ``market_data`` dict on each call.

    This is the primary entry point that codec agents use.
    """
    with _default_lock:
        return _default_computer.compute(pair, price, volume)


def reset_default_computer() -> None:
    """Reset the module-level singleton (useful in tests)."""
    global _default_computer
    _default_computer = IndicatorComputer(buffer_size=200)
