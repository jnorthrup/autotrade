package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 18: VWAP Mean Reversion
 *
 * Ported from Python codec_18_vwap_mean_reversion.py (non-MLX fallback).
 * Volume-weighted anchor reversion strategy.
 */
public class Codec18VwapMeanReversion extends BaseCodecExpert {

    private final double deviationThreshold;

    public Codec18VwapMeanReversion() {
        super(18, "vwap_mean_reversion", "1.0");
        this.deviationThreshold = 0.01;
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double vwap = getDouble(marketData, "vwap", price);
        double volume = getDouble(marketData, "volume", 0.0);
        double avgVolume = getDouble(marketData, "avg_volume", volume);

        double direction = 0.0;
        double confidence = 0.2;

        double vwapDev = 0.0;
        if (vwap > 0) {
            double deviation = (price - vwap) / vwap;
            vwapDev = deviation;

            if (Math.abs(deviation) > deviationThreshold) {
                direction = -Math.signum(deviation) * Math.min(Math.abs(deviation) * 20.0, 1.0);
                confidence = Math.min(Math.abs(deviation) * 30.0 + 0.3, 1.0);
            }

            double volRatio = avgVolume > 0 ? volume / avgVolume : 1.0;
            if (volRatio > 1.5) {
                confidence = Math.min(confidence * 1.2, 1.0);
            }
        }

        // regime_label defaults to 1 in Python (not in our data)
        // Python: if regime == 1: confidence *= 1.2 else confidence *= 0.7
        // regime_label not available -> defaults to 1
        confidence *= 1.2;

        // Non-MLX: no model override

        recordInstruments("vwap_dev", vwapDev);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
