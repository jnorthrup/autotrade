package com.xtrade.kline;

import java.util.Objects;

/**
 * Unique identity for a kline stream.
 */
public final class KlineSeriesId {
    private final String venue;
    private final String symbol;
    private final KlineInterval interval;

    public KlineSeriesId(String venue, String symbol, KlineInterval interval) {
        this.venue = requireText(venue, "venue");
        this.symbol = requireText(symbol, "symbol");
        this.interval = Objects.requireNonNull(interval, "interval must not be null");
    }

    public String venue() {
        return venue;
    }

    public String symbol() {
        return symbol;
    }

    public KlineInterval interval() {
        return interval;
    }

    public String cacheKey() {
        return venue + ":" + symbol + ":" + interval.wireName();
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof KlineSeriesId)) {
            return false;
        }
        KlineSeriesId that = (KlineSeriesId) other;
        return venue.equals(that.venue) && symbol.equals(that.symbol) && interval == that.interval;
    }

    @Override
    public int hashCode() {
        return Objects.hash(venue, symbol, interval);
    }

    @Override
    public String toString() {
        return cacheKey();
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
