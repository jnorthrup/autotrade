package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 17: ADX Trend Strength
 *
 * Ported from Python codec_17_adx_trend_strength.py (non-MLX fallback).
 * Directional movement + ADX power filter.
 */
public class Codec17AdxTrendStrength extends BaseCodecExpert {

    private final double adxThreshold;

    public Codec17AdxTrendStrength() {
        super(17, "adx_trend_strength", "1.0");
        this.adxThreshold = 25.0;
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double adxVal = getDouble(marketData, "adx",
                getDouble(marketData, "adx_14", 0.0));
        double plusDi = getDouble(marketData, "plus_di", 0.0);
        double minusDi = getDouble(marketData, "minus_di", 0.0);

        double diDiff = plusDi - minusDi;
        double direction = Math.tanh(diDiff / 20.0);

        double adxFactor;
        if (adxVal >= adxThreshold) {
            adxFactor = Math.min(1.0, adxVal / 50.0 + 0.3);
        } else if (adxVal >= 15) {
            adxFactor = 0.15 + (adxVal - 15) / 10.0 * 0.3;
        } else {
            adxFactor = 0.1 + adxVal / 15.0 * 0.1;
        }

        double confidence = Math.min(1.0, adxFactor + Math.abs(diDiff) / 50.0);

        double momentum = getDouble(marketData, "momentum",
                getDouble(marketData, "log_return", 0.0));
        if (Math.signum(direction) == Math.signum(momentum) && adxVal > 20) {
            confidence = Math.min(confidence * 1.2, 1.0);
        }

        // Non-MLX: no model override

        recordInstruments("adx", adxVal);
        recordInstruments("plus_di", plusDi);
        recordInstruments("minus_di", minusDi);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
