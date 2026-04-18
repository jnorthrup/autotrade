package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 01: Volatility Breakout
 *
 * Ported from Python codec_01_volatility_breakout.py (non-MLX fallback).
 * Captures explosive moves when volatility expands beyond normal ranges.
 */
public class Codec01VolatilityBreakout extends BaseCodecExpert {

    private final double atrMultiplier;
    private final int lookback;

    public Codec01VolatilityBreakout() {
        super(1, "volatility_breakout", "1.0");
        this.atrMultiplier = 2.0;
        this.lookback = 20;
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double high = getDouble(marketData, "high", price);
        double low = getDouble(marketData, "low", price);
        double atr = getDouble(marketData, "atr_14", (high - low) * 0.5);

        double volatilitySignal = 0.0;
        if (atr > 0) {
            double rangeExpansion = (high - low) / atr;
            if (rangeExpansion > atrMultiplier) {
                double momentum = getDouble(marketData, "momentum", 0.0);
                volatilitySignal = Math.signum(momentum) * Math.min(rangeExpansion / 3.0, 1.0);
            }
        }

        // Non-MLX fallback (matches Python when HAS_MLX is False)
        double confidence = Math.abs(volatilitySignal) + 0.3;
        double direction = volatilitySignal;

        recordInstruments("volatility_signal", volatilitySignal);
        recordInstruments("atr_norm", getDouble(marketData, "atr_14", 0.0) / (price > 0 ? price : 1.0));
        recordInstruments("momentum", getDouble(marketData, "momentum", 0.0));

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(confidence, direction);
    }
}
