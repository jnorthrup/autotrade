package com.xtrade.kline;

import java.util.Objects;

/**
 * Historical request for a single kline stream.
 */
public final class KlineBatchRequest {
    private final KlineSeriesId seriesId;
    private final Long startTimeMillisInclusive;
    private final Long endTimeMillisExclusive;
    private final Integer limit;
    private final boolean closedBarsOnly;

    private KlineBatchRequest(KlineSeriesId seriesId,
                              Long startTimeMillisInclusive,
                              Long endTimeMillisExclusive,
                              Integer limit,
                              boolean closedBarsOnly) {
        this.seriesId = Objects.requireNonNull(seriesId, "seriesId must not be null");
        this.startTimeMillisInclusive = startTimeMillisInclusive;
        this.endTimeMillisExclusive = endTimeMillisExclusive;
        this.limit = limit;
        this.closedBarsOnly = closedBarsOnly;
        validate();
    }

    public static KlineBatchRequest between(KlineSeriesId seriesId, long startTimeMillisInclusive, long endTimeMillisExclusive) {
        return new KlineBatchRequest(seriesId, startTimeMillisInclusive, endTimeMillisExclusive, null, true);
    }

    public static KlineBatchRequest latest(KlineSeriesId seriesId, int limit) {
        return new KlineBatchRequest(seriesId, null, null, limit, true);
    }

    public KlineBatchRequest withClosedBarsOnly(boolean closedBarsOnly) {
        return new KlineBatchRequest(seriesId, startTimeMillisInclusive, endTimeMillisExclusive, limit, closedBarsOnly);
    }

    public KlineSeriesId seriesId() {
        return seriesId;
    }

    public Long startTimeMillisInclusive() {
        return startTimeMillisInclusive;
    }

    public Long endTimeMillisExclusive() {
        return endTimeMillisExclusive;
    }

    public Integer limit() {
        return limit;
    }

    public boolean closedBarsOnly() {
        return closedBarsOnly;
    }

    private void validate() {
        if (startTimeMillisInclusive != null && startTimeMillisInclusive < 0) {
            throw new IllegalArgumentException("startTimeMillisInclusive must not be negative");
        }
        if (endTimeMillisExclusive != null && endTimeMillisExclusive < 0) {
            throw new IllegalArgumentException("endTimeMillisExclusive must not be negative");
        }
        if (startTimeMillisInclusive != null && endTimeMillisExclusive != null
                && endTimeMillisExclusive <= startTimeMillisInclusive) {
            throw new IllegalArgumentException("endTimeMillisExclusive must be greater than startTimeMillisInclusive");
        }
        if (limit != null && limit <= 0) {
            throw new IllegalArgumentException("limit must be positive");
        }
        if (startTimeMillisInclusive == null && endTimeMillisExclusive != null) {
            throw new IllegalArgumentException("endTimeMillisExclusive requires startTimeMillisInclusive");
        }
        if (startTimeMillisInclusive == null && limit == null) {
            throw new IllegalArgumentException("request must define either a [start,end) range or a latest limit");
        }
    }
}
