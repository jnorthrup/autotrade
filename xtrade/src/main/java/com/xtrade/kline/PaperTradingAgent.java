package com.xtrade.kline;

import com.xtrade.PaperTradingEngine;

/**
 * Agent SPI for driving paper trades directly from live or replayed kline bars.
 */
public interface PaperTradingAgent {
    default void onConnected(KlineSeriesId seriesId, PaperTradingEngine engine) {
        // default no-op
    }

    void onBar(KlineBar bar, PaperTradingEngine engine);

    default void onDisconnected(KlineSeriesId seriesId, PaperTradingEngine engine) {
        // default no-op
    }
}
