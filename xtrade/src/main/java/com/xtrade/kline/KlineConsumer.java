package com.xtrade.kline;

import java.util.List;

/**
 * Consumer SPI for historical replay and live kline updates.
 */
public interface KlineConsumer {
    default void onBackfill(KlineSeriesId seriesId, List<KlineBar> bars) {
        // default no-op
    }

    void onLiveBar(KlineBar bar);

    default void onError(KlineSeriesId seriesId, Exception error) {
        // default no-op
    }

    default void onClosed(KlineSeriesId seriesId) {
        // default no-op
    }
}
