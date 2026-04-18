package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 05: Pairs Trading
 *
 * Statistical arbitrage between correlated instruments.
 */
public class Codec05PairsTrading extends BaseCodecExpert {

    public Codec05PairsTrading() {
        super(5, "pairs_trading", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double sma20 = getDouble(marketData, "sma_20", price);
        double rsi = getDouble(marketData, "rsi_14", 50.0);

        double zscore = sma20 > 0 ? (price - sma20) / sma20 : 0.0;
        double direction = zscore > 0 ? -1.0 : 1.0;
        double conviction = Math.min(1.0, Math.abs(zscore) * 5.0);

        recordInstruments("zscore", zscore);
        recordInstruments("rsi_14", rsi);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
