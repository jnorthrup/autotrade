package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 13: RSI Reversal
 *
 * Ported from Python codec_13_rsi_reversal.py (non-MLX fallback).
 * RSI-based mean reversion strategy.
 */
public class Codec13RsiReversal extends BaseCodecExpert {

    private final double oversold;
    private final double overbought;

    public Codec13RsiReversal() {
        super(13, "rsi_reversal", "1.0");
        this.oversold = 40.0;
        this.overbought = 60.0;
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double rsi = getDouble(marketData, "rsi_14", 50.0);

        double direction = 0.0;
        double confidence = 0.35;

        if (rsi < oversold) {
            direction = (oversold - rsi) / oversold;
            confidence = Math.min(1.0 - rsi / oversold + 0.3, 1.0);
        } else if (rsi > overbought) {
            direction = -(rsi - overbought) / (100.0 - overbought);
            confidence = Math.min((rsi - overbought) / (100.0 - overbought) + 0.3, 1.0);
        } else {
            // In the neutral zone, use RSI deviation from 50 as a weak signal
            double deviation = (rsi - 50.0) / 50.0;
            direction = -deviation * 0.3;
            confidence = 0.35 + Math.abs(deviation) * 0.15;
        }

        // Stochastic confirmation
        double stoch = getDouble(marketData, "stoch_k", 50.0);
        if (stoch < 30 && rsi < oversold) {
            direction *= 1.3;
            confidence = Math.min(confidence * 1.2, 1.0);
        } else if (stoch > 70 && rsi > overbought) {
            direction *= 1.3;
            confidence = Math.min(confidence * 1.2, 1.0);
        }

        // Non-MLX: no model override

        recordInstruments("rsi_14", rsi);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
