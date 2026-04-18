package com.xtrade.codec;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;

/**
 * Tests for BaseCodecExpert: verifies OB memory ring buffer,
 * trade ledger, instrument bus, and signal validation.
 */
public class BaseCodecExpertTest {

    private static class TestCodec extends BaseCodecExpert {
        TestCodec() { super(99, "test_codec", "1.0"); }

        @Override
        public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
            return validateSignal(0.5, 0.7);
        }
    }

    @Test
    public void testObMemoryDimensions() {
        TestCodec codec = new TestCodec();
        assertEquals(512, codec.obMemory.length);
        assertEquals(64, codec.obMemory[0].length);
    }

    @Test
    public void testObMemoryRingBuffer() {
        TestCodec codec = new TestCodec();
        for (int i = 0; i < 600; i++) {
            double[] vec = new double[64];
            vec[0] = i;
            codec.updateObMemory(1.0, vec);
        }
        assertEquals(88, codec.obMemoryIdx);
        assertEquals(599.0, codec.obMemory[87][0], 0.001);
    }

    @Test
    public void testObMemoryPadding() {
        TestCodec codec = new TestCodec();
        double[] shortVec = new double[10];
        shortVec[3] = 42.0;
        codec.updateObMemory(1.0, shortVec);
        assertEquals(42.0, codec.obMemory[0][3], 0.001);
        assertEquals(0.0, codec.obMemory[0][63], 0.001);
    }

    @Test
    public void testObMemoryTruncation() {
        TestCodec codec = new TestCodec();
        double[] longVec = new double[100];
        longVec[63] = 99.0;
        longVec[99] = 77.0;
        codec.updateObMemory(1.0, longVec);
        assertEquals(99.0, codec.obMemory[0][63], 0.001);
    }

    @Test
    public void testValidateSignalClips() {
        TestCodec codec = new TestCodec();
        SignalResult r1 = codec.validateSignal(1.5, -2.0);
        assertEquals(1.0, r1.getConviction(), 0.001);
        assertEquals(-1.0, r1.getDirection(), 0.001);

        SignalResult r2 = codec.validateSignal(-0.5, 0.0);
        assertEquals(0.0, r2.getConviction(), 0.001);
        assertEquals(0.0, r2.getDirection(), 0.001);
    }

    @Test
    public void testTradeLedger() {
        TestCodec codec = new TestCodec();
        for (int i = 0; i < 5; i++) {
            codec.recordTradeOutcome(10.0, 1.0, 0.01);
        }
        Map<String, Object> ledger = codec.getTradeLedger();
        assertEquals(50.0, (Double) ledger.get("cumulative_pnl"), 0.001);
        assertEquals(5, ledger.get("signal_count"));
        assertEquals(1.0, (Double) ledger.get("hit_rate"), 0.001);

        codec.recordTradeOutcome(-5.0, -1.0, -0.01);
        ledger = codec.getTradeLedger();
        assertEquals(45.0, (Double) ledger.get("cumulative_pnl"), 0.001);
        assertEquals(6, ledger.get("signal_count"));
        assertEquals((5.0 / 6.0), (Double) ledger.get("hit_rate"), 0.001);
    }

    @Test
    public void testTradeLedgerSharpeEstimate() {
        TestCodec codec = new TestCodec();
        for (int i = 0; i < 9; i++) {
            codec.recordTradeOutcome(1.0, 1.0, 0.01);
        }
        Map<String, Object> ledger = codec.getTradeLedger();
        assertEquals(0.0, (Double) ledger.get("sharpe"), 0.001);

        codec.recordTradeOutcome(1.0, 1.0, 0.01);
        ledger = codec.getTradeLedger();
        assertEquals(1.0, (Double) ledger.get("sharpe"), 0.001);
    }

    @Test
    public void testTradeLedgerReset() {
        TestCodec codec = new TestCodec();
        codec.recordTradeOutcome(10.0, 1.0, 0.01);
        codec.resetTradeLedger();
        Map<String, Object> ledger = codec.getTradeLedger();
        assertEquals(0.0, (Double) ledger.get("cumulative_pnl"), 0.001);
        assertEquals(0, ledger.get("signal_count"));
    }

    @Test
    public void testInstrumentsBus() {
        TestCodec codec = new TestCodec();
        codec.recordInstruments("rsi_14", 65.3);
        codec.recordInstruments("macd_hist", -0.002);
        assertEquals(65.3, codec.getInstruments().get("rsi_14"), 0.001);
        assertEquals(-0.002, codec.getInstruments().get("macd_hist"), 0.001);
    }

    @Test
    public void testResetRuntimeState() {
        TestCodec codec = new TestCodec();
        double[] vec = new double[64];
        vec[0] = 42.0;
        codec.updateObMemory(1.0, vec);
        codec.recordInstruments("test", 1.0);
        codec.resetRuntimeState();
        assertEquals(0, codec.obMemoryIdx);
        assertEquals(0.0, codec.obMemory[0][0], 0.001);
        assertTrue(codec.getInstruments().isEmpty());
    }

    @Test
    public void testObSummary() {
        TestCodec codec = new TestCodec();
        assertTrue(codec.getObSummary().isEmpty());

        double[] vec = new double[64];
        for (int i = 0; i < 64; i++) vec[i] = 1.0;
        codec.updateObMemory(1.0, vec);
        Map<String, Double> summary = codec.getObSummary();
        assertEquals(1.0, summary.get("bar_count"), 0.001);
        assertEquals(1.0, summary.get("mean"), 0.001);
        assertEquals(0.0, summary.get("std"), 0.001);
    }

    @Test
    public void testForwardReturnsValidSignal() {
        TestCodec codec = new TestCodec();
        SignalResult result = codec.forward(null, new double[0]);
        assertEquals(0.5, result.getConviction(), 0.001);
        assertEquals(0.7, result.getDirection(), 0.001);
    }
}
