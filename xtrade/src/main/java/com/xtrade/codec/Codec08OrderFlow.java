package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 08: Order Flow
 *
 * Trades based on order book imbalance signals.
 */
public class Codec08OrderFlow extends BaseCodecExpert {

    public Codec08OrderFlow() {
        super(8, "order_flow", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double volume = getDouble(marketData, "volume", 0.0);
        double sma20 = getDouble(marketData, "sma_20", price);
        double logReturn = getDouble(marketData, "log_return", 0.0);

        double direction = logReturn > 0 ? 1.0 : -1.0;
        double conviction = Math.min(1.0, Math.abs(logReturn) * 100.0);

        double priceVsSma = sma20 > 0 ? (price - sma20) / sma20 : 0.0;
        if (Math.abs(priceVsSma) > 0.01) {
            conviction = Math.min(1.0, conviction * 0.7 + Math.abs(priceVsSma) * 30.0 * 0.3);
        }

        recordInstruments("log_return", logReturn);
        recordInstruments("price_vs_sma", priceVsSma);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
