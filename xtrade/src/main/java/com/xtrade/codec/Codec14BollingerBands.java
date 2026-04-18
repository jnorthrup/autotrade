package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 14: Bollinger Bands
 *
 * Ported from Python codec_14_bollinger_bands.py (non-MLX fallback).
 * Bollinger Band mean reversion and breakout.
 */
public class Codec14BollingerBands extends BaseCodecExpert {

    public Codec14BollingerBands() {
        super(14, "bollinger_bands", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double bbUpper = getDouble(marketData, "bb_upper", price);
        double bbLower = getDouble(marketData, "bb_lower", price);
        double bbMid = getDouble(marketData, "bb_mid", price);

        double direction = 0.0;
        double confidence = 0.2;

        if (bbUpper > bbLower) {
            double bbWidth = bbUpper - bbLower;
            double bbPosition = bbWidth > 0 ? (price - bbLower) / bbWidth : 0.5;

            if (bbPosition < 0) {
                direction = 0.8;
                confidence = Math.min(Math.abs(bbPosition) + 0.4, 1.0);
            } else if (bbPosition > 1) {
                direction = -0.8;
                confidence = Math.min(bbPosition - 1 + 0.4, 1.0);
            } else if (bbPosition < 0.2) {
                direction = 0.5;
                confidence = 0.5;
            } else if (bbPosition > 0.8) {
                direction = -0.5;
                confidence = 0.5;
            }
        }

        double sma15 = getDouble(marketData, "sma_15", price);
        double bandwidth = sma15 > 0 ? (bbUpper - bbLower) / sma15 : 0.0;
        if (bandwidth < 0.02) {
            confidence *= 0.5;
        }

        // Non-MLX: no model override

        double bbWidth = bbUpper - bbLower;
        double bbPos = bbWidth > 0 ? (price - bbLower) / bbWidth : 0.5;
        recordInstruments("bb_pct", bbPos);
        recordInstruments("bb_width", bandwidth);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
