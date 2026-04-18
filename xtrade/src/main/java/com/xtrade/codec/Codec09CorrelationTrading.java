package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 09: Correlation Trading
 *
 * Exploits correlations between instruments for spread trading.
 */
public class Codec09CorrelationTrading extends BaseCodecExpert {

    public Codec09CorrelationTrading() {
        super(9, "correlation_trading", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double sma15 = getDouble(marketData, "sma_15", price);
        double ema12 = getDouble(marketData, "ema_12", price);
        double rsi = getDouble(marketData, "rsi_14", 50.0);

        double spread = sma15 > 0 ? (price - sma15) / sma15 : 0.0;
        double direction = spread > 0 ? -1.0 : 1.0;
        double conviction = Math.min(1.0, Math.abs(spread) * 20.0);

        if (rsi > 70) conviction *= 1.2;
        else if (rsi < 30) conviction *= 1.2;

        recordInstruments("spread", spread);
        recordInstruments("rsi_14", rsi);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
