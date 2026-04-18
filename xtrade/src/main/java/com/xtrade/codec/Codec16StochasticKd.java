package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 16: Stochastic KD
 *
 * Ported from Python codec_16_stochastic_kd.py (non-MLX fallback).
 * %K/%D crossover with overbought/oversold filter.
 */
public class Codec16StochasticKd extends BaseCodecExpert {

    private final double oversold;
    private final double overbought;

    public Codec16StochasticKd() {
        super(16, "stochastic_kd", "1.0");
        this.oversold = 20.0;
        this.overbought = 80.0;
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double stochK = getDouble(marketData, "stoch_k", 50.0);
        double stochD = getDouble(marketData, "stoch_d", 50.0);

        double direction = 0.0;
        double confidence = 0.2;

        double crossSignal = stochK - stochD;

        if (stochK < oversold && stochD < oversold) {
            if (crossSignal > 0) {
                direction = Math.min(crossSignal / 10.0 + 0.5, 1.0);
                confidence = 0.7 + (oversold - stochK) / 100.0;
            }
        } else if (stochK > overbought && stochD > overbought) {
            if (crossSignal < 0) {
                direction = Math.max(crossSignal / 10.0 - 0.5, -1.0);
                confidence = 0.7 + (stochK - overbought) / 100.0;
            }
        } else {
            direction = Math.signum(crossSignal) * Math.min(Math.abs(crossSignal) / 10.0, 0.5);
            confidence = 0.4;
        }

        // Non-MLX: no model override

        recordInstruments("stoch_k", stochK);
        recordInstruments("stoch_d", stochD);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
