package com.xtrade.codec;

import java.util.Map;

/**
 * Abstract base class for all 24 codec expert implementations.
 *
 * Mirrors the Python BaseExpert with:
 * - ob_memory[512][64] ring buffer for temporal order book data
 * - trade ledger for performance tracking
 * - instruments dict (instrument predictor bus)
 * - forward(), updateObMemory(), validateSignal() matching Python API
 */
public abstract class BaseCodecExpert implements CodecStrategy {

    protected static final int OB_MEMORY_SIZE = 512;
    protected static final int OB_MEMORY_DIM = 64;

    protected final String name;
    protected final String version;
    protected final int codecId;
    protected final String strategyName;

    // Temporal order book buffer: 512-bar ring buffer, 64-dim feature per bar
    protected final double[][] obMemory;
    protected int obMemoryIdx;

    // Instrument predictor bus: named indicator readings
    protected final Map<String, Double> instruments;

    // Trade performance ledger
    protected final TradeLedger tradeLedger;

    protected BaseCodecExpert(int codecId, String strategyName, String version) {
        this.codecId = codecId;
        this.strategyName = strategyName;
        this.name = String.format("codec_%02d", codecId);
        this.version = version != null ? version : "1.0";

        this.obMemory = new double[OB_MEMORY_SIZE][OB_MEMORY_DIM];
        this.obMemoryIdx = 0;

        this.instruments = new java.util.LinkedHashMap<>();
        this.tradeLedger = new TradeLedger();
    }

    @Override
    public int getCodecId() {
        return codecId;
    }

    @Override
    public String getStrategyName() {
        return strategyName;
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
    }

    // ── OB Memory Ring Buffer ─────────────────────────────────────────

    /**
     * Update the 512-bar temporal order book ring buffer.
     * Mirrors Python BaseExpert.update_ob_memory().
     *
     * @param direction    trade direction signal
     * @param indicatorVec feature vector for current bar
     */
    public void updateObMemory(double direction, double[] indicatorVec) {
        double[] row = obMemory[obMemoryIdx];
        if (indicatorVec.length != OB_MEMORY_DIM) {
            // Reset row to zeros first
            java.util.Arrays.fill(row, 0.0);
            int copyLen = Math.min(indicatorVec.length, OB_MEMORY_DIM);
            System.arraycopy(indicatorVec, 0, row, 0, copyLen);
        } else {
            System.arraycopy(indicatorVec, 0, row, 0, OB_MEMORY_DIM);
        }
        obMemoryIdx = (obMemoryIdx + 1) % OB_MEMORY_SIZE;
    }

    /**
     * Reset per-stream transient state.
     */
    public void resetRuntimeState() {
        for (double[] row : obMemory) {
            java.util.Arrays.fill(row, 0.0);
        }
        obMemoryIdx = 0;
        instruments.clear();
    }

    /**
     * Summary statistics of the temporal order book buffer.
     */
    public Map<String, Double> getObSummary() {
        Map<String, Double> summary = new java.util.LinkedHashMap<>();
        if (obMemoryIdx == 0) {
            return summary;
        }
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < obMemoryIdx; i++) {
            for (int j = 0; j < OB_MEMORY_DIM; j++) {
                sum += obMemory[i][j];
                count++;
            }
        }
        double mean = count > 0 ? sum / count : 0.0;
        double varSum = 0.0;
        for (int i = 0; i < obMemoryIdx; i++) {
            for (int j = 0; j < OB_MEMORY_DIM; j++) {
                double d = obMemory[i][j] - mean;
                varSum += d * d;
            }
        }
        double std = count > 0 ? Math.sqrt(varSum / count) : 0.0;
        summary.put("mean", mean);
        summary.put("std", std);
        summary.put("bar_count", (double) obMemoryIdx);
        return summary;
    }

    // ── Instrument Predictor Bus ──────────────────────────────────────

    /**
     * Populate instruments dict with named indicator readings.
     * Mirrors Python BaseExpert.record_instruments().
     */
    public void recordInstruments(Map<String, Double> values) {
        instruments.putAll(values);
    }

    public void recordInstruments(String key, double value) {
        instruments.put(key, value);
    }

    public Map<String, Double> getInstruments() {
        return instruments;
    }

    // ── Trade Ledger ──────────────────────────────────────────────────

    public void recordTradeOutcome(double realizedPnl, double direction, double actualReturn) {
        tradeLedger.recordTradeOutcome(realizedPnl, direction, actualReturn);
    }

    public Map<String, Object> getTradeLedger() {
        return tradeLedger.getLedger();
    }

    public void resetTradeLedger() {
        tradeLedger.reset();
    }

    // ── Signal Validation ─────────────────────────────────────────────

    /**
     * Clip and validate signal output to valid ranges.
     * Mirrors Python BaseExpert.validate_signal().
     */
    public SignalResult validateSignal(double conviction, double direction) {
        conviction = Math.max(0.0, Math.min(1.0, conviction));
        direction = Math.max(-1.0, Math.min(1.0, direction));
        return new SignalResult(conviction, direction);
    }

    // ── Utility helpers for subclass codec strategies ────────────────

    /**
     * Get a double value from a Map<String, Object> marketData dict.
     */
    protected static double getDouble(Map<String, Object> map, String key, double defaultValue) {
        Object val = map.get(key);
        if (val == null) return defaultValue;
        if (val instanceof Number) return ((Number) val).doubleValue();
        return defaultValue;
    }

    /**
     * Pad or truncate an indicator vector to exactly OB_MEMORY_DIM (64) elements.
     */
    protected static double[] padOrTruncate(double[] indicatorVec) {
        if (indicatorVec == null) return new double[OB_MEMORY_DIM];
        if (indicatorVec.length == OB_MEMORY_DIM) return indicatorVec;
        double[] result = new double[OB_MEMORY_DIM];
        int copyLen = Math.min(indicatorVec.length, OB_MEMORY_DIM);
        System.arraycopy(indicatorVec, 0, result, 0, copyLen);
        return result;
    }

    @Override
    public String toString() {
        return name + " (v" + version + ")";
    }
}
