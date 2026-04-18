package com.xtrade.codec;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;

/**
 * Cross-validation tests for IndicatorComputer against Python reference values.
 * Python values generated from showdown/indicators.py with the same price/volume sequence.
 */
public class IndicatorComputerTest {

    private static final double EPSILON = 1e-6;

    private static final double[] PRICES = {
        100.0, 101.0, 99.5, 102.0, 103.5, 101.5, 100.0, 98.0, 97.5, 99.0,
        100.5, 102.0, 103.0, 104.5, 105.0, 106.0, 105.5, 104.0, 103.0, 102.5,
        101.0, 100.0, 99.0, 98.5, 97.0, 96.5, 97.5, 98.0, 99.5, 100.5
    };
    private static final double VOLUME = 1000.0;

    @Test
    public void testSma15() {
        Map<String, Object> result = feedAll();
        assertEquals(100.56666666666666, d(result, "sma_15"), EPSILON);
    }

    @Test
    public void testSma20() {
        Map<String, Object> result = feedAll();
        assertEquals(101.175, d(result, "sma_20"), EPSILON);
    }

    @Test
    public void testEma12() {
        Map<String, Object> result = feedAll();
        assertEquals(99.54650131357909, d(result, "ema_12"), EPSILON);
    }

    @Test
    public void testEma26() {
        Map<String, Object> result = feedAll();
        assertEquals(100.22979737121986, d(result, "ema_26"), EPSILON);
    }

    @Test
    public void testMacd() {
        Map<String, Object> result = feedAll();
        assertEquals(-0.6832960576407743, d(result, "macd"), EPSILON);
    }

    @Test
    public void testMacdSignal() {
        Map<String, Object> result = feedAll();
        assertEquals(-0.4253042943254652, d(result, "macd_signal"), EPSILON);
    }

    @Test
    public void testMacdHist() {
        Map<String, Object> result = feedAll();
        assertEquals(-0.2579917633153091, d(result, "macd_hist"), EPSILON);
    }

    @Test
    public void testRsi14() {
        Map<String, Object> result = feedAll();
        assertEquals(51.562250844424476, d(result, "rsi_14"), EPSILON);
    }

    @Test
    public void testBollingerBands() {
        Map<String, Object> result = feedAll();
        assertEquals(101.175, d(result, "bb_mid"), EPSILON);
        assertEquals(106.91324886180445, d(result, "bb_upper"), EPSILON);
        assertEquals(95.43675113819555, d(result, "bb_lower"), EPSILON);
    }

    @Test
    public void testAtr14() {
        Map<String, Object> result = feedAll();
        assertEquals(1.1209462225987998, d(result, "atr_14"), EPSILON);
    }

    @Test
    public void testStochastic() {
        Map<String, Object> result = feedAll();
        assertEquals(44.44444444444444, d(result, "stoch_k"), EPSILON);
        assertEquals(30.60428849902534, d(result, "stoch_d"), EPSILON);
    }

    @Test
    public void testAdx() {
        Map<String, Object> result = feedAll();
        assertEquals(14.791665111086413, d(result, "adx_14"), EPSILON);
        assertEquals(51.562250844424476, d(result, "plus_di"), EPSILON);
        assertEquals(48.43774915557554, d(result, "minus_di"), EPSILON);
    }

    @Test
    public void testVwap() {
        Map<String, Object> result = feedAll();
        assertEquals(100.85, d(result, "vwap"), EPSILON);
    }

    @Test
    public void testMomentum() {
        Map<String, Object> result = feedAll();
        assertEquals(-1.9512195121951237, d(result, "momentum"), EPSILON);
    }

    @Test
    public void testAvgVolume() {
        Map<String, Object> result = feedAll();
        assertEquals(1000.0, d(result, "avg_volume"), EPSILON);
    }

    @Test
    public void testLogReturn() {
        Map<String, Object> result = feedAll();
        assertEquals(0.010000083334583399, d(result, "log_return"), EPSILON);
    }

    @Test
    public void testIntermediateStep25() {
        IndicatorComputer ic = new IndicatorComputer(200);
        Map<String, Object> result = null;
        for (int i = 0; i < 26; i++) {
            result = ic.compute("BTC/USD", PRICES[i], VOLUME);
        }
        assertEquals(101.83333333333333, d(result, "sma_15"), EPSILON);
        assertEquals(101.125, d(result, "sma_20"), EPSILON);
        assertEquals(99.97654695834521, d(result, "ema_12"), EPSILON);
        assertEquals(100.68170430658166, d(result, "ema_26"), EPSILON);
        assertEquals(-0.7051573482364546, d(result, "macd"), EPSILON);
        assertEquals(0.11997316737788807, d(result, "macd_signal"), EPSILON);
        assertEquals(-0.8251305156143427, d(result, "macd_hist"), EPSILON);
        assertEquals(37.014662851255466, d(result, "rsi_14"), EPSILON);
        assertEquals(1.1594926678812187, d(result, "atr_14"), EPSILON);
        assertEquals(0.0, d(result, "stoch_k"), EPSILON);
        assertEquals(0.0, d(result, "stoch_d"), EPSILON);
        assertEquals(16.69755917250303, d(result, "adx_14"), EPSILON);
        assertEquals(37.01466285125547, d(result, "plus_di"), EPSILON);
        assertEquals(62.98533714874455, d(result, "minus_di"), EPSILON);
        assertEquals(101.15384615384616, d(result, "vwap"), EPSILON);
        assertEquals(-8.9622641509434, d(result, "momentum"), EPSILON);
    }

    @Test
    public void testSingleTick() {
        IndicatorComputer ic = new IndicatorComputer(200);
        Map<String, Object> result = ic.compute("BTC/USD", 100.0, 1000.0);
        assertEquals(100.0, d(result, "price"), EPSILON);
        assertEquals(100.0, d(result, "sma_15"), EPSILON);
        assertEquals(100.0, d(result, "ema_12"), EPSILON);
        assertEquals(50.0, d(result, "rsi_14"), EPSILON);
        assertEquals(0.0, d(result, "macd_hist"), EPSILON);
        assertEquals(0.0, d(result, "momentum"), EPSILON);
        assertEquals(0.0, d(result, "atr_14"), EPSILON);
    }

    @Test
    public void testAllSamePrice() {
        IndicatorComputer ic = new IndicatorComputer(200);
        Map<String, Object> result = null;
        for (int i = 0; i < 30; i++) {
            result = ic.compute("BTC/USD", 100.0, 1000.0);
        }
        assertEquals(100.0, d(result, "sma_15"), EPSILON);
        assertEquals(100.0, d(result, "ema_12"), EPSILON);
        assertEquals(100.0, d(result, "bb_mid"), EPSILON);
        assertEquals(100.0, d(result, "bb_upper"), EPSILON);
        assertEquals(100.0, d(result, "bb_lower"), EPSILON);
        assertEquals(0.0, d(result, "momentum"), EPSILON);
        assertEquals(0.0, d(result, "log_return"), EPSILON);
        assertEquals(100.0, d(result, "rsi_14"), EPSILON);
    }

    @Test
    public void testPerPairIsolation() {
        IndicatorComputer ic = new IndicatorComputer(200);
        ic.compute("BTC/USD", 100.0, 1000.0);
        ic.compute("ETH/USD", 200.0, 500.0);
        Map<String, Object> btcResult = ic.compute("BTC/USD", 101.0, 1000.0);
        Map<String, Object> ethResult = ic.compute("ETH/USD", 201.0, 500.0);
        assertEquals(101.0, d(btcResult, "price"), EPSILON);
        assertEquals(201.0, d(ethResult, "price"), EPSILON);
    }

    private Map<String, Object> feedAll() {
        IndicatorComputer ic = new IndicatorComputer(200);
        Map<String, Object> result = null;
        for (double price : PRICES) {
            result = ic.compute("BTC/USD", price, VOLUME);
        }
        return result;
    }

    private static double d(Map<String, Object> map, String key) {
        Object val = map.get(key);
        if (val instanceof Number) return ((Number) val).doubleValue();
        return 0.0;
    }
}
