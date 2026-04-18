package com.xtrade.showdown;

/**
 * Data source interface for the ShowdownHarness.
 * Produces a sequence of TickData objects that are fed identically to all agents.
 */
public interface IDataSource {

    /**
     * Returns true if there is more tick data available.
     */
    boolean hasNext();

    /**
     * Returns the next tick of market data.
     *
     * @return the next TickData, never null
     * @throws java.util.NoSuchElementException if no more data
     */
    TickData next();

    /**
     * Reset the data source to its initial state.
     */
    void reset();
}
