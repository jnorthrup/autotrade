package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 20: Hurst Regime
 *
 * Uses Hurst exponent estimation to detect trending vs mean-reverting regimes.
 */
public class Codec20HurstRegime extends BaseCodecExpert {

    public Codec20HurstRegime() {
        super(20, "hurst_regime", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double momentum = getDouble(marketData, "momentum", 0.0);
        double adx = getDouble(marketData, "adx_14", 0.0);
        double sma20 = getDouble(marketData, "sma_20", price);

        double hurstEst = 0.5;
        if (adx > 30.0) {
            hurstEst = 0.6 + Math.min(0.3, (adx - 30.0) / 100.0);
        } else if (adx < 15.0) {
            hurstEst = 0.4 - Math.min(0.2, (15.0 - adx) / 100.0);
        }

        double direction = 0.0;
        double conviction = 0.0;

        if (hurstEst > 0.55) {
            direction = momentum > 0 ? 1.0 : -1.0;
            conviction = Math.min(1.0, (hurstEst - 0.5) * 4.0);
        } else if (hurstEst < 0.45) {
            direction = momentum > 0 ? -1.0 : 1.0;
            conviction = Math.min(1.0, (0.5 - hurstEst) * 4.0);
        }

        recordInstruments("hurst_est", hurstEst);
        recordInstruments("adx", adx);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
