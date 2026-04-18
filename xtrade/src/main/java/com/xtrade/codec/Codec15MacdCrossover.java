package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 15: MACD Crossover
 *
 * Ported from Python codec_15_macd_crossover.py (non-MLX fallback).
 * Signal-line and histogram divergence detection.
 */
public class Codec15MacdCrossover extends BaseCodecExpert {

    public Codec15MacdCrossover() {
        super(15, "macd_crossover", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double macd = getDouble(marketData, "macd", 0.0);
        double macdSignal = getDouble(marketData, "macd_signal", 0.0);
        double macdHist = getDouble(marketData, "macd_hist", macd - macdSignal);

        double direction = 0.0;
        double confidence = 0.2;

        if (macdHist > 0) {
            direction = Math.min(macdHist * 20.0, 1.0);
            confidence = Math.min(Math.abs(macdHist) * 10.0 + 0.3, 1.0);
        } else if (macdHist < 0) {
            direction = Math.max(macdHist * 20.0, -1.0);
            confidence = Math.min(Math.abs(macdHist) * 10.0 + 0.3, 1.0);
        }

        double momentum = getDouble(marketData, "momentum", 0.0);
        if (Math.signum(direction) == Math.signum(momentum)) {
            confidence *= 1.2;
        }

        // Non-MLX: no model override

        recordInstruments("macd_hist", macdHist);
        recordInstruments("macd_line", macd);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
