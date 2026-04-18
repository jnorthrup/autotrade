package com.xtrade.codec;

import com.xtrade.Signal;
import com.xtrade.TradingPair;
import org.knowm.xchange.dto.marketdata.Ticker;

import java.util.Map;

/**
 * Extended strategy interface for codec expert models.
 *
 * In addition to the base TradingStrategy evaluate(), codec strategies
 * support forward(marketData, indicatorVec) returning a SignalResult
 * with conviction and direction, mirroring the Python BaseExpert.forward().
 */
public interface CodecStrategy extends com.xtrade.TradingStrategy {

    /**
     * Generate trade signal from market data and indicator vector.
     *
     * @param marketData    dict-like container with price, volume, and computed indicators
     * @param indicatorVec  computed technical indicator vector (double[])
     * @return SignalResult with conviction ∈ [0,1] and direction ∈ [-1,1]
     */
    SignalResult forward(Map<String, Object> marketData, double[] indicatorVec);

    /**
     * Return the codec ID (1-24).
     */
    int getCodecId();

    /**
     * Return the strategy-specific name (e.g. "volatility_breakout").
     */
    String getStrategyName();

    /**
     * Default implementation of TradingStrategy.evaluate() that delegates
     * to forward() with empty market data / indicator vec.
     */
    @Override
    default Signal evaluate(TradingPair pair, Map<TradingPair, Ticker> tickers) {
        Ticker ticker = tickers.get(pair);
        if (ticker == null) {
            return Signal.HOLD;
        }
        Map<String, Object> marketData = new java.util.HashMap<>();
        marketData.put("price", ticker.getLast().doubleValue());
        marketData.put("volume", ticker.getVolume().doubleValue());
        SignalResult result = forward(marketData, new double[0]);
        if (result.getConviction() < 0.3) {
            return Signal.HOLD;
        }
        return result.getDirection() > 0 ? Signal.BUY : Signal.SELL;
    }

    @Override
    default String getName() {
        return getStrategyName();
    }

    @Override
    default void reset() {
        // subclasses override as needed
    }
}
