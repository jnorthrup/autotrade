package com.xtrade.kline;

import java.util.Objects;

/**
 * Producer connectivity snapshot exported by the kline feed.
 */
public final class KlineFeedProducerStatus {
    private final String producerId;
    private final String producerName;
    private final int publishedSeriesCount;
    private final long lastPublishTimeMillis;
    private final long lastPublishAgeMillis;
    private final long publishedBars;
    private final boolean connected;
    private final boolean stale;

    public KlineFeedProducerStatus(String producerId,
                                   String producerName,
                                   int publishedSeriesCount,
                                   long lastPublishTimeMillis,
                                   long lastPublishAgeMillis,
                                   long publishedBars,
                                   boolean connected,
                                   boolean stale) {
        this.producerId = requireText(producerId, "producerId");
        this.producerName = requireText(producerName, "producerName");
        this.publishedSeriesCount = publishedSeriesCount;
        this.lastPublishTimeMillis = lastPublishTimeMillis;
        this.lastPublishAgeMillis = lastPublishAgeMillis;
        this.publishedBars = publishedBars;
        this.connected = connected;
        this.stale = stale;
    }

    public String producerId() {
        return producerId;
    }

    public String producerName() {
        return producerName;
    }

    public int publishedSeriesCount() {
        return publishedSeriesCount;
    }

    public long lastPublishTimeMillis() {
        return lastPublishTimeMillis;
    }

    public long lastPublishAgeMillis() {
        return lastPublishAgeMillis;
    }

    public long publishedBars() {
        return publishedBars;
    }

    public boolean connected() {
        return connected;
    }

    public boolean stale() {
        return stale;
    }

    private static String requireText(String value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return trimmed;
    }
}
