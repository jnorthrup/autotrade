package com.xtrade.codec;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Unit tests for the 12+ core codec strategies.
 *
 * Each test feeds a fixed price series through IndicatorComputer to produce
 * deterministic indicator values, then passes those indicators to the codec's
 * forward() method and verifies that:
 *   - conviction is in [0, 1]
 *   - direction is in [-1, 1]
 *   - output is deterministic (same inputs => same outputs)
 */
public class CodecStrategyTest {

    private static final double EPSILON = 1e-9;

    // Fixed price/volume series used to populate IndicatorComputer deterministically
    private static final double[] PRICES = {
        100.0, 101.0, 99.5, 102.0, 103.5, 101.5, 100.0, 98.0, 97.5, 99.0,
        100.5, 102.0, 103.0, 104.5, 105.0, 106.0, 105.5, 104.0, 103.0, 102.5,
        101.0, 100.0, 99.0, 98.5, 97.0, 96.5, 97.5, 98.0, 99.5, 100.5
    };
    private static final double VOLUME = 1000.0;

    /** Market data computed once from the fixed price series. */
    private Map<String, Object> marketData;

    @BeforeEach
    public void setUp() {
        IndicatorComputer ic = new IndicatorComputer(200);
        Map<String, Object> result = null;
        for (double price : PRICES) {
            result = ic.compute("BTC/USD", price, VOLUME);
        }
        this.marketData = result;
    }

    // ── Helper ──────────────────────────────────────────────────────

    private void assertValidSignal(SignalResult r) {
        assertNotNull(r, "SignalResult must not be null");
        assertTrue(r.getConviction() >= 0.0 && r.getConviction() <= 1.0,
                "conviction must be in [0,1], got " + r.getConviction());
        assertTrue(r.getDirection() >= -1.0 && r.getDirection() <= 1.0,
                "direction must be in [-1,1], got " + r.getDirection());
    }

    private void assertDeterministic(BaseCodecExpert codec) {
        double[] emptyVec = new double[0];
        // Reset stateful codecs
        codec.resetRuntimeState();
        SignalResult r1 = codec.forward(marketData, emptyVec);
        codec.resetRuntimeState();
        SignalResult r2 = codec.forward(marketData, emptyVec);
        assertEquals(r1.getConviction(), r2.getConviction(), EPSILON,
                codec.getStrategyName() + " conviction must be deterministic");
        assertEquals(r1.getDirection(), r2.getDirection(), EPSILON,
                codec.getStrategyName() + " direction must be deterministic");
    }

    // ── Codec 01: Volatility Breakout ──────────────────────────────

    @Test
    public void testCodec01VolatilityBreakout() {
        Codec01VolatilityBreakout codec = new Codec01VolatilityBreakout();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        // With our price series producing positive BB width, we expect a non-zero signal
        double bbWidth = BaseCodecExpert.getDouble(marketData, "bb_upper", 0.0)
                - BaseCodecExpert.getDouble(marketData, "bb_lower", 0.0);
        if (bbWidth > 0) {
            assertTrue(r.getConviction() > 0.0, "Expected positive conviction with non-zero BB width");
        }
        assertDeterministic(codec);
    }

    // ── Codec 02: Momentum Trend ───────────────────────────────────

    @Test
    public void testCodec02MomentumTrend() {
        Codec02MomentumTrend codec = new Codec02MomentumTrend();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        // Momentum is negative (-1.95%), so direction should be negative
        double momentum = BaseCodecExpert.getDouble(marketData, "momentum", 0.0);
        if (Math.abs(momentum) > 0.5) {
            assertEquals(momentum > 0 ? 1 : -1, r.getDirection() > 0 ? 1 : -1,
                    "Direction should match momentum sign");
        }
        assertDeterministic(codec);
    }

    // ── Codec 03: Mean Reversion ───────────────────────────────────

    @Test
    public void testCodec03MeanReversion() {
        Codec03MeanReversion codec = new Codec03MeanReversion();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 04: Trend Following ──────────────────────────────────

    @Test
    public void testCodec04TrendFollowing() {
        Codec04TrendFollowing codec = new Codec04TrendFollowing();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 05: Pairs Trading ────────────────────────────────────

    @Test
    public void testCodec05PairsTrading() {
        Codec05PairsTrading codec = new Codec05PairsTrading();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 06: Grid Trading ─────────────────────────────────────

    @Test
    public void testCodec06GridTrading() {
        Codec06GridTrading codec = new Codec06GridTrading();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 07: Volume Profile ───────────────────────────────────

    @Test
    public void testCodec07VolumeProfile() {
        Codec07VolumeProfile codec = new Codec07VolumeProfile();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 08: Order Flow ───────────────────────────────────────

    @Test
    public void testCodec08OrderFlow() {
        Codec08OrderFlow codec = new Codec08OrderFlow();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 09: Correlation Trading ──────────────────────────────

    @Test
    public void testCodec09CorrelationTrading() {
        Codec09CorrelationTrading codec = new Codec09CorrelationTrading();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 10: Liquidity Making ─────────────────────────────────

    @Test
    public void testCodec10LiquidityMaking() {
        Codec10LiquidityMaking codec = new Codec10LiquidityMaking();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 11: Sector Rotation ──────────────────────────────────

    @Test
    public void testCodec11SectorRotation() {
        Codec11SectorRotation codec = new Codec11SectorRotation();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 12: Composite Alpha ──────────────────────────────────

    @Test
    public void testCodec12CompositeAlpha() {
        Codec12CompositeAlpha codec = new Codec12CompositeAlpha();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 13: RSI Reversal ─────────────────────────────────────

    @Test
    public void testCodec13RsiReversal() {
        Codec13RsiReversal codec = new Codec13RsiReversal();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        // RSI ~51.5 is in neutral zone, conviction should be low
        assertDeterministic(codec);
    }

    // ── Codec 14: Bollinger Bands ──────────────────────────────────

    @Test
    public void testCodec14BollingerBands() {
        Codec14BollingerBands codec = new Codec14BollingerBands();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 15: MACD Crossover ───────────────────────────────────

    @Test
    public void testCodec15MacdCrossover() {
        Codec15MacdCrossover codec = new Codec15MacdCrossover();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        // MACD histogram is negative, so direction should be -1
        double macdHist = BaseCodecExpert.getDouble(marketData, "macd_hist", 0.0);
        if (macdHist < 0) {
            assertTrue(r.getDirection() < 0, "Direction should be negative when MACD hist is negative");
        }
        assertDeterministic(codec);
    }

    // ── Codec 16: Stochastic K/D ───────────────────────────────────

    @Test
    public void testCodec16StochasticKd() {
        Codec16StochasticKd codec = new Codec16StochasticKd();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 17: ADX Trend Strength ───────────────────────────────

    @Test
    public void testCodec17AdxTrendStrength() {
        Codec17AdxTrendStrength codec = new Codec17AdxTrendStrength();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 18: VWAP Mean Reversion ──────────────────────────────

    @Test
    public void testCodec18VwapMeanReversion() {
        Codec18VwapMeanReversion codec = new Codec18VwapMeanReversion();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 19: Kalman Filter Trend ──────────────────────────────

    @Test
    public void testCodec19KalmanFilterTrend() {
        Codec19KalmanFilterTrend codec = new Codec19KalmanFilterTrend();
        // Kalman filter is stateful: feed the series forward
        IndicatorComputer ic = new IndicatorComputer(200);
        SignalResult lastResult = null;
        for (double price : PRICES) {
            Map<String, Object> md = ic.compute("BTC/USD", price, VOLUME);
            lastResult = codec.forward(md, new double[0]);
        }
        assertNotNull(lastResult);
        assertTrue(lastResult.getConviction() >= 0.0 && lastResult.getConviction() <= 1.0,
                "Kalman conviction in [0,1]");
        assertTrue(lastResult.getDirection() >= -1.0 && lastResult.getDirection() <= 1.0,
                "Kalman direction in [-1,1]");
    }

    @Test
    public void testCodec19KalmanDeterministic() {
        // Feed same series twice to a fresh codec, verify same final result
        Codec19KalmanFilterTrend codec1 = new Codec19KalmanFilterTrend();
        Codec19KalmanFilterTrend codec2 = new Codec19KalmanFilterTrend();
        IndicatorComputer ic1 = new IndicatorComputer(200);
        IndicatorComputer ic2 = new IndicatorComputer(200);
        SignalResult r1 = null, r2 = null;
        for (double price : PRICES) {
            r1 = codec1.forward(ic1.compute("BTC/USD", price, VOLUME), new double[0]);
            r2 = codec2.forward(ic2.compute("BTC/USD", price, VOLUME), new double[0]);
        }
        assertEquals(r1.getConviction(), r2.getConviction(), EPSILON);
        assertEquals(r1.getDirection(), r2.getDirection(), EPSILON);
    }

    // ── Codec 20: Hurst Regime ─────────────────────────────────────

    @Test
    public void testCodec20HurstRegime() {
        Codec20HurstRegime codec = new Codec20HurstRegime();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 21: Random Forest Classifier (ML heuristic) ──────────

    @Test
    public void testCodec21RandomForestClassifier() {
        Codec21RandomForestClassifier codec = new Codec21RandomForestClassifier();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 22: XGBoost Signal (ML heuristic) ────────────────────

    @Test
    public void testCodec22XgboostSignal() {
        Codec22XgboostSignal codec = new Codec22XgboostSignal();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 23: Transformer Attention (ML heuristic) ─────────────

    @Test
    public void testCodec23TransformerAttention() {
        Codec23TransformerAttention codec = new Codec23TransformerAttention();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── Codec 24: Z-Score Stat Arb ─────────────────────────────────

    @Test
    public void testCodec24ZscoreStatArb() {
        Codec24ZscoreStatArb codec = new Codec24ZscoreStatArb();
        SignalResult r = codec.forward(marketData, new double[0]);
        assertValidSignal(r);
        assertDeterministic(codec);
    }

    // ── CodecFactory.createExpert(id) for all 24 ───────────────────

    @Test
    public void testCodecFactoryCreateExpertAll24() {
        for (int i = 1; i <= 24; i++) {
            BaseCodecExpert expert = CodecFactory.createExpert(i);
            assertNotNull(expert, "CodecFactory.createExpert(" + i + ") should return non-null");
            assertEquals(i, expert.getCodecId(), "Codec ID mismatch for id=" + i);
            // Each factory-created expert should produce a valid signal
            SignalResult r = expert.forward(marketData, new double[0]);
            assertNotNull(r, "forward() should return non-null for codec " + i);
            assertTrue(r.getConviction() >= 0.0 && r.getConviction() <= 1.0,
                    "Codec " + i + " conviction out of range: " + r.getConviction());
            assertTrue(r.getDirection() >= -1.0 && r.getDirection() <= 1.0,
                    "Codec " + i + " direction out of range: " + r.getDirection());
        }
    }

    @Test
    public void testCodecFactoryInvalidIdThrows() {
        assertThrows(IllegalArgumentException.class, () -> CodecFactory.createExpert(0));
        assertThrows(IllegalArgumentException.class, () -> CodecFactory.createExpert(25));
        assertThrows(IllegalArgumentException.class, () -> CodecFactory.createExpert(-1));
    }

    // ── All codecs produce valid signals with minimal data ─────────

    @Test
    public void testAllCodecsWithMinimalData() {
        IndicatorComputer ic = new IndicatorComputer(200);
        Map<String, Object> minimalData = ic.compute("BTC/USD", 100.0, 500.0);

        for (int i = 1; i <= 24; i++) {
            BaseCodecExpert expert = CodecFactory.createExpert(i);
            SignalResult r = expert.forward(minimalData, new double[0]);
            assertNotNull(r, "Codec " + i + " forward() returned null with minimal data");
            assertTrue(r.getConviction() >= 0.0 && r.getConviction() <= 1.0,
                    "Codec " + i + " conviction out of range with minimal data: " + r.getConviction());
            assertTrue(r.getDirection() >= -1.0 && r.getDirection() <= 1.0,
                    "Codec " + i + " direction out of range with minimal data: " + r.getDirection());
        }
    }

    // ── All codecs produce valid signals with empty data ───────────

    @Test
    public void testAllCodecsWithEmptyData() {
        Map<String, Object> emptyData = new HashMap<>();
        for (int i = 1; i <= 24; i++) {
            BaseCodecExpert expert = CodecFactory.createExpert(i);
            SignalResult r = expert.forward(emptyData, new double[0]);
            assertNotNull(r, "Codec " + i + " forward() returned null with empty data");
            assertTrue(r.getConviction() >= 0.0 && r.getConviction() <= 1.0,
                    "Codec " + i + " conviction out of range with empty data: " + r.getConviction());
            assertTrue(r.getDirection() >= -1.0 && r.getDirection() <= 1.0,
                    "Codec " + i + " direction out of range with empty data: " + r.getDirection());
        }
    }
}
