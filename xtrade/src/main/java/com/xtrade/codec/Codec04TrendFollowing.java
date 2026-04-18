package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 04: Trend Following
 *
 * Ported from Python codec_04_trend_following.py (non-MLX fallback).
 * Classic trend following with multi-timeframe confirmation.
 */
public class Codec04TrendFollowing extends BaseCodecExpert {

    public Codec04TrendFollowing() {
        super(4, "trend_following", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double sma15 = getDouble(marketData, "sma_15", price);
        double sma20 = getDouble(marketData, "sma_20", price);

        double trendScore = 0;
        // Compare price vs SMA(15) and SMA(15) vs SMA(20) for trend confirmation
        if (price > sma15) {
            trendScore += 1;
        } else {
            trendScore -= 1;
        }
        if (sma15 > sma20) {
            trendScore += 1;
        } else {
            trendScore -= 1;
        }
        if (price > sma20) {
            trendScore += 0.5;
        } else {
            trendScore -= 0.5;
        }

        double adx = getDouble(marketData, "adx_14", 0.0);
        double trendStrength = adx > 0 ? adx / 100.0 : 0.0;

        double direction = trendScore / 2.5;
        double confidence = Math.min(trendStrength + 0.3, 1.0);

        // Non-MLX: no model override

        recordInstruments("momentum", getDouble(marketData, "momentum", 0.0));

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
