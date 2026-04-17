package com.xtrade.kline;

/**
 * Origin channel for a canonical kline bar.
 */
public enum KlineSource {
    REST_BACKFILL,
    WEBSOCKET_MUX,
    FILE_REPLAY,
    SYNTHETIC_AGGREGATION,
    MANUAL_IMPORT
}
