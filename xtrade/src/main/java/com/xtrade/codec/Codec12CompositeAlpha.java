package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 12: Composite Alpha
 *
 * Combines multiple alpha signals into a composite score.
 */
public class Codec12CompositeAlpha extends BaseCodecExpert {

    public Codec12CompositeAlpha() {
        super(12, "composite_alpha", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double rsi = getDouble(marketData, "rsi_14", 50.0);
        double macdHist = getDouble(marketData, "macd_hist", 0.0);
        double momentum = getDouble(marketData, "momentum", 0.0);
        double adx = getDouble(marketData, "adx_14", 0.0);

        double rsiScore = (rsi - 50.0) / 50.0;
        double macdScore = macdHist > 0 ? 1.0 : -1.0;
        double momScore = momentum > 0 ? 1.0 : -1.0;

        double composite = rsiScore * 0.3 + macdScore * 0.3 + momScore * 0.4;
        double direction = composite > 0 ? 1.0 : -1.0;
        double conviction = Math.min(1.0, Math.abs(composite) * (adx > 20 ? 1.0 : 0.5));

        recordInstruments("composite_score", composite);
        recordInstruments("rsi_score", rsiScore);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
