package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 03: Mean Reversion
 *
 * Ported from Python codec_03_mean_reversion.py (non-MLX fallback).
 * Bets on price returning to mean after extreme moves.
 */
public class Codec03MeanReversion extends BaseCodecExpert {

    private final int lookback;
    private final double zThreshold;

    public Codec03MeanReversion() {
        super(3, "mean_reversion", "1.0");
        this.lookback = 20;
        this.zThreshold = 2.0;
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double sma20 = getDouble(marketData, "sma_15", price);
        double rollingStd = price * 0.02;  // vol_5m not available, default

        double reversionSignal = 0.0;
        if (sma20 > 0 && rollingStd > 0) {
            double zScore = (price - sma20) / rollingStd;
            if (Math.abs(zScore) > zThreshold) {
                reversionSignal = -Math.signum(zScore) * Math.min(Math.abs(zScore) / 4.0, 1.0);
            }
        }

        double rsi = getDouble(marketData, "rsi_14", 50.0);
        double rsiSignal = 0.0;
        if (rsi > 70) {
            rsiSignal = -0.5;
        } else if (rsi < 30) {
            rsiSignal = 0.5;
        }

        double combined = reversionSignal * 0.6 + rsiSignal * 0.4;

        // Non-MLX fallback
        double confidence = Math.abs(combined) + 0.2;
        double direction = combined;

        if (indicatorVec.length > 0) {
            recordInstruments("returns_last", indicatorVec[0]);
        }

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
