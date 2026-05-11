import unittest
import numpy as np
from showdown.indicators import (
    _sma, _ema, _rolling_ema_series, _true_range,
    _BarBuffer, IndicatorComputer,
    compute_market_data, reset_default_computer
)

class TestIndicators(unittest.TestCase):

    # -- Core helper tests --

    def test_sma(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        self.assertEqual(_sma(arr, 3), 4.0)
        self.assertEqual(_sma(arr, 5), 3.0)
        self.assertEqual(_sma(arr, 10), 5.0)  # period > len(arr), returns last value
        self.assertEqual(_sma(np.array([]), 3), 0.0)
        self.assertEqual(_sma(np.array([10]), 3), 10.0)

    def test_ema(self):
        arr = np.array([10, 20, 30], dtype=np.float64)
        # k = 2 / (2+1) = 2/3
        # ema0 = 10
        # ema1 = 20 * (2/3) + 10 * (1/3) = 40/3 + 10/3 = 50/3 = 16.666...
        # ema2 = 30 * (2/3) + (50/3) * (1/3) = 60/3 + 50/9 = 180/9 + 50/9 = 230/9 = 25.555...
        expected = 230/9
        self.assertAlmostEqual(_ema(arr, 2), expected)
        self.assertEqual(_ema(np.array([]), 2), 0.0)
        self.assertEqual(_ema(np.array([10]), 2), 10.0)

    def test_rolling_ema_series(self):
        arr = np.array([10, 20, 30], dtype=np.float64)
        series = _rolling_ema_series(arr, 2)
        self.assertEqual(len(series), 3)
        self.assertEqual(series[0], 10.0)
        self.assertAlmostEqual(series[1], 50/3)
        self.assertAlmostEqual(series[2], 230/9)

        empty_series = _rolling_ema_series(np.array([]), 2)
        self.assertEqual(len(empty_series), 0)

    def test_true_range(self):
        highs = np.array([10, 12, 11], dtype=np.float64)
        lows = np.array([8, 9, 7], dtype=np.float64)
        closes = np.array([9, 11, 8], dtype=np.float64)

        # tr1: max(12-9, abs(12-9), abs(9-9)) = max(3, 3, 0) = 3
        # tr2: max(11-7, abs(11-11), abs(7-11)) = max(4, 0, 4) = 4
        tr = _true_range(highs, lows, closes)
        np.testing.assert_array_almost_equal(tr, [3.0, 4.0])

        self.assertEqual(len(_true_range(np.array([10]), np.array([8]), np.array([9]))), 0)

    # -- _BarBuffer tests --

    def test_bar_buffer_circular(self):
        buf = _BarBuffer(maxlen=3)
        buf.append_tick(10.0, 100.0)
        buf.append_tick(11.0, 110.0)
        buf.append_tick(12.0, 120.0)

        o, h, l, c, v, t = buf.as_arrays()
        self.assertEqual(len(c), 3)
        np.testing.assert_array_equal(c, [10.0, 11.0, 12.0])

        buf.append_tick(13.0, 130.0)
        o, h, l, c, v, t = buf.as_arrays()
        self.assertEqual(len(c), 3)
        np.testing.assert_array_equal(c, [11.0, 12.0, 13.0])
        np.testing.assert_array_equal(v, [110.0, 120.0, 130.0])

    def test_bar_buffer_empty(self):
        buf = _BarBuffer(maxlen=3)
        o, h, l, c, v, t = buf.as_arrays()
        self.assertEqual(len(c), 0)

    # -- IndicatorComputer static methods tests --

    def test_rsi(self):
        # Flat prices: _rsi returns 100.0 when avg_loss is 0.0
        closes = np.full(20, 100.0)
        self.assertEqual(IndicatorComputer._rsi(closes, 14), 100.0)

        # Rising prices
        closes = np.array([100 + i for i in range(20)], dtype=np.float64)
        self.assertEqual(IndicatorComputer._rsi(closes, 14), 100.0)

        # Insufficient data
        self.assertEqual(IndicatorComputer._rsi(np.array([100, 101]), 14), 50.0)

    def test_atr(self):
        highs = np.array([10, 12, 11, 13], dtype=np.float64)
        lows = np.array([8, 9, 7, 10], dtype=np.float64)
        closes = np.array([9, 11, 8, 12], dtype=np.float64)
        # tr: [max(12-9, 12-9, 9-9)=3, max(11-7, 11-11, 7-11)=4, max(13-10, 13-8, 10-8)=5]
        # tr: [3, 4, 5]
        # mean(tr) = 4
        self.assertEqual(IndicatorComputer._atr(highs, lows, closes, 14), 4.0)

    def test_stochastic(self):
        highs = np.array([110]*20, dtype=np.float64)
        lows = np.array([90]*20, dtype=np.float64)
        closes = np.array([100]*20, dtype=np.float64)
        # k = (100-90)/(110-90) * 100 = 10/20 * 100 = 50
        k, d = IndicatorComputer._stochastic(highs, lows, closes, 14, 3)
        self.assertEqual(k, 50.0)
        self.assertEqual(d, 50.0)

    def test_adx(self):
        # Minimal data
        highs = np.array([10, 11], dtype=np.float64)
        lows = np.array([9, 8], dtype=np.float64)
        closes = np.array([9.5, 10], dtype=np.float64)
        adx, pdi, mdi = IndicatorComputer._adx(highs, lows, closes, 14)
        self.assertEqual(adx, 0.0)

        # More data (still simple cases)
        highs = np.full(30, 10.0)
        lows = np.full(30, 8.0)
        closes = np.full(30, 9.0)
        adx, pdi, mdi = IndicatorComputer._adx(highs, lows, closes, 14)
        self.assertEqual(adx, 0.0) # No movement

    def test_vwap(self):
        typical = np.array([10, 20], dtype=np.float64)
        volumes = np.array([100, 200], dtype=np.float64)
        # (10*100 + 20*200) / (100 + 200) = (1000 + 4000) / 300 = 5000 / 300 = 16.666...
        self.assertAlmostEqual(IndicatorComputer._vwap(typical, volumes), 50/3)

        self.assertEqual(IndicatorComputer._vwap(np.array([]), np.array([])), 0.0)

    def test_momentum(self):
        closes = np.array([10, 12, 15, 11, 10], dtype=np.float64)
        # period 2: (10 / 15 - 1) * 100 = (0.666 - 1) * 100 = -33.333...
        self.assertAlmostEqual(IndicatorComputer._momentum(closes, 2), -100/3)

        # Insufficient data
        self.assertEqual(IndicatorComputer._momentum(closes, 10), 0.0)

    # -- Module-level tests --

    def test_compute_market_data(self):
        reset_default_computer()
        res1 = compute_market_data("BTC/USDT", 40000.0, 1.0)
        self.assertEqual(res1["price"], 40000.0)
        self.assertEqual(res1["sma_15"], 40000.0)

        res2 = compute_market_data("BTC/USDT", 41000.0, 2.0)
        self.assertEqual(res2["price"], 41000.0)
        # In indicators.py, _sma returns last price if len(arr) < period
        self.assertEqual(res2["sma_15"], 41000.0)

    def test_reset_default_computer(self):
        compute_market_data("BTC/USDT", 40000.0, 1.0)
        reset_default_computer()
        # After reset, the first call should behave as if there's no history
        res = compute_market_data("BTC/USDT", 41000.0, 1.0)
        self.assertEqual(res["sma_15"], 41000.0)

if __name__ == "__main__":
    unittest.main()
