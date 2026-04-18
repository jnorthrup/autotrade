package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 02: Momentum Trend
 *
 * Ported from Python codec_02_momentum_trend.py (non-MLX fallback).
 * Follows established trends with momentum confirmation.
 */
public class Codec02MomentumTrend extends BaseCodecExpert {

    public Codec02MomentumTrend() {
        super(2, "momentum_trend", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double ema12 = getDouble(marketData, "ema_12", price);
        double ema26 = getDouble(marketData, "ema_26", price);
        double sma20 = getDouble(marketData, "sma_20", price);

        double trendSignal = 0.0;
        // Use available EMAs: compare fast (ema12) vs slow (ema26) and price position
        double trendAlignment = 0;
        if (ema12 > ema26) {
            trendAlignment += 1;
        } else {
            trendAlignment -= 1;
        }
        if (price > sma20) {
            trendAlignment += 1;
        } else {
            trendAlignment -= 1;
        }

        double momentum = getDouble(marketData, "momentum", 0.0);
        trendSignal = (trendAlignment / 2.0) * 0.7 + Math.signum(momentum) * 0.3;

        // Non-MLX fallback
        double confidence = Math.abs(trendSignal) + 0.25;
        double direction = trendSignal;

        recordInstruments("momentum_fast", momentum);
        if (indicatorVec.length > 0) {
            recordInstruments("returns_last", indicatorVec[0]);
        }

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
