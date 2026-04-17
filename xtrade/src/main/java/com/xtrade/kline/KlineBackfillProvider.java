package com.xtrade.kline;

import java.util.List;

/**
 * Producer-side backfill SPI used by the draw-thru cache when a consumer requests history
 * that is missing from memory.
 */
public interface KlineBackfillProvider {
    List<KlineBar> load(KlineBatchRequest request);
}
