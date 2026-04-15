package com.xtrade;

import org.knowm.xchange.dto.marketdata.Ticker;

import java.util.Map;

/**
 * Evaluates market data for a trading pair and produces a {@link Signal}.
 * <p>
 * Implementations may maintain internal state (e.g. moving averages, price
 * history) across calls. Implementations must be thread-safe if shared
 * across concurrent trading loops.
 */
public interface TradingStrategy {

    /**
     * Returns the human-readable name of this strategy.
     */
    String getName();

    /**
     * Evaluates the strategy for a single trading pair given the current
     * set of ticker snapshots.
     *
     * @param pair    the pair to evaluate
     * @param tickers current ticker data for all monitored pairs (may not contain every pair)
     * @return a non-null Signal (BUY, SELL, or HOLD)
     */
    Signal evaluate(TradingPair pair, Map<TradingPair, Ticker> tickers);

    /**
     * Resets any internal state (price histories, etc.).
     */
    void reset();
}
