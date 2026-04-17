package com.xtrade.kline;

import java.util.Objects;

/**
 * Subscription request that optionally replays historical bars before live forwarding begins.
 */
public final class KlineSubscriptionRequest {
    private final KlineSeriesId seriesId;
    private final KlineBatchRequest backfillRequest;
    private final boolean replayBackfill;
    private final boolean closedBarsOnly;

    private KlineSubscriptionRequest(KlineSeriesId seriesId,
                                     KlineBatchRequest backfillRequest,
                                     boolean replayBackfill,
                                     boolean closedBarsOnly) {
        this.seriesId = Objects.requireNonNull(seriesId, "seriesId must not be null");
        this.backfillRequest = backfillRequest;
        this.replayBackfill = replayBackfill;
        this.closedBarsOnly = closedBarsOnly;
    }

    public static KlineSubscriptionRequest liveOnly(KlineSeriesId seriesId) {
        return new KlineSubscriptionRequest(seriesId, null, false, true);
    }

    public static KlineSubscriptionRequest backfillThenLive(KlineBatchRequest backfillRequest) {
        return new KlineSubscriptionRequest(backfillRequest.seriesId(), backfillRequest, true, backfillRequest.closedBarsOnly());
    }

    public KlineSubscriptionRequest withClosedBarsOnly(boolean closedBarsOnly) {
        KlineBatchRequest adjusted = backfillRequest == null ? null : backfillRequest.withClosedBarsOnly(closedBarsOnly);
        return new KlineSubscriptionRequest(seriesId, adjusted, replayBackfill, closedBarsOnly);
    }

    public KlineSeriesId seriesId() {
        return seriesId;
    }

    public KlineBatchRequest backfillRequest() {
        return backfillRequest;
    }

    public boolean replayBackfill() {
        return replayBackfill;
    }

    public boolean closedBarsOnly() {
        return closedBarsOnly;
    }
}
