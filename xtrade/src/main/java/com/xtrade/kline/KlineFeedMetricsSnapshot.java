package com.xtrade.kline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Immutable operational metrics snapshot for the draw-thru feed.
 */
public final class KlineFeedMetricsSnapshot {
    private final long generatedAtMillis;
    private final long staleAfterMillis;
    private final long cacheRequests;
    private final long cacheHits;
    private final long cacheMisses;
    private final double cacheHitRate;
    private final long backfillRequests;
    private final long backfillBarsLoaded;
    private final long publishedBars;
    private final double averageFeedLatencyMillis;
    private final long maxFeedLatencyMillis;
    private final long lastFeedLatencyMillis;
    private final int seriesCount;
    private final long bufferedBars;
    private final long activeSubscriptions;
    private final int maxBarsPerSeries;
    private final List<KlineFeedProducerStatus> producers;

    public KlineFeedMetricsSnapshot(long generatedAtMillis,
                                    long staleAfterMillis,
                                    long cacheRequests,
                                    long cacheHits,
                                    long cacheMisses,
                                    double cacheHitRate,
                                    long backfillRequests,
                                    long backfillBarsLoaded,
                                    long publishedBars,
                                    double averageFeedLatencyMillis,
                                    long maxFeedLatencyMillis,
                                    long lastFeedLatencyMillis,
                                    int seriesCount,
                                    long bufferedBars,
                                    long activeSubscriptions,
                                    int maxBarsPerSeries,
                                    List<KlineFeedProducerStatus> producers) {
        this.generatedAtMillis = generatedAtMillis;
        this.staleAfterMillis = staleAfterMillis;
        this.cacheRequests = cacheRequests;
        this.cacheHits = cacheHits;
        this.cacheMisses = cacheMisses;
        this.cacheHitRate = cacheHitRate;
        this.backfillRequests = backfillRequests;
        this.backfillBarsLoaded = backfillBarsLoaded;
        this.publishedBars = publishedBars;
        this.averageFeedLatencyMillis = averageFeedLatencyMillis;
        this.maxFeedLatencyMillis = maxFeedLatencyMillis;
        this.lastFeedLatencyMillis = lastFeedLatencyMillis;
        this.seriesCount = seriesCount;
        this.bufferedBars = bufferedBars;
        this.activeSubscriptions = activeSubscriptions;
        this.maxBarsPerSeries = maxBarsPerSeries;
        this.producers = Collections.unmodifiableList(new ArrayList<>(producers));
    }

    public long generatedAtMillis() { return generatedAtMillis; }
    public long staleAfterMillis() { return staleAfterMillis; }
    public long cacheRequests() { return cacheRequests; }
    public long cacheHits() { return cacheHits; }
    public long cacheMisses() { return cacheMisses; }
    public double cacheHitRate() { return cacheHitRate; }
    public long backfillRequests() { return backfillRequests; }
    public long backfillBarsLoaded() { return backfillBarsLoaded; }
    public long publishedBars() { return publishedBars; }
    public double averageFeedLatencyMillis() { return averageFeedLatencyMillis; }
    public long maxFeedLatencyMillis() { return maxFeedLatencyMillis; }
    public long lastFeedLatencyMillis() { return lastFeedLatencyMillis; }
    public int seriesCount() { return seriesCount; }
    public long bufferedBars() { return bufferedBars; }
    public long activeSubscriptions() { return activeSubscriptions; }
    public int maxBarsPerSeries() { return maxBarsPerSeries; }
    public List<KlineFeedProducerStatus> producers() { return producers; }
}
