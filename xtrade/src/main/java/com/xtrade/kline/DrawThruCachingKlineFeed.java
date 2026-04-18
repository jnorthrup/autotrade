package com.xtrade.kline;

import java.time.Clock;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-memory draw-thru caching feed that buffers live bars, backfills on demand,
 * forwards subsequent updates to active subscribers, and exposes operational
 * health plus metrics snapshots.
 */
public final class DrawThruCachingKlineFeed implements DrawThruKlineFeed {
    private final int maxBarsPerSeries;
    private final Clock clock;
    private final Map<String, ProducerHandleImpl> producers = new ConcurrentHashMap<>();
    private final Map<String, ProducerState> producerStates = new ConcurrentHashMap<>();
    private final Map<KlineSeriesId, SeriesBuffer> buffers = new ConcurrentHashMap<>();
    private final Map<KlineSeriesId, CopyOnWriteArrayList<SubscriptionImpl>> subscriptions = new ConcurrentHashMap<>();
    private final AtomicLong subscriptionSequence = new AtomicLong(1L);
    private final AtomicLong requestCount = new AtomicLong();
    private final AtomicLong cacheHitCount = new AtomicLong();
    private final AtomicLong cacheMissCount = new AtomicLong();
    private final AtomicLong backfillRequestCount = new AtomicLong();
    private final AtomicLong backfillBarsLoaded = new AtomicLong();
    private final AtomicLong publishedBarCount = new AtomicLong();
    private final AtomicLong totalFeedLatencyMillis = new AtomicLong();
    private final AtomicLong maxFeedLatencyMillis = new AtomicLong();
    private final AtomicLong lastFeedLatencyMillis = new AtomicLong();

    public DrawThruCachingKlineFeed(int maxBarsPerSeries) {
        this(maxBarsPerSeries, Clock.systemUTC());
    }

    public DrawThruCachingKlineFeed(int maxBarsPerSeries, Clock clock) {
        if (maxBarsPerSeries <= 0) {
            throw new IllegalArgumentException("maxBarsPerSeries must be positive");
        }
        this.maxBarsPerSeries = maxBarsPerSeries;
        this.clock = Objects.requireNonNull(clock, "clock must not be null");
    }

    @Override
    public KlineProducerHandle registerProducer(KlineProducerRegistration registration) {
        Objects.requireNonNull(registration, "registration must not be null");
        ProducerHandleImpl handle = new ProducerHandleImpl(registration);
        ProducerHandleImpl prior = producers.putIfAbsent(registration.producerId(), handle);
        if (prior != null) {
            throw new IllegalArgumentException("producer already registered: " + registration.producerId());
        }
        producerStates.put(registration.producerId(), new ProducerState(registration, clock.millis()));
        return handle;
    }

    @Override
    public List<KlineBar> requestBars(KlineBatchRequest request) {
        Objects.requireNonNull(request, "request must not be null");
        requestCount.incrementAndGet();
        SeriesBuffer buffer = bufferFor(request.seriesId());
        List<KlineBar> cached = buffer.select(request);
        if (isSatisfied(request, cached, buffer)) {
            cacheHitCount.incrementAndGet();
            return cached;
        }
        cacheMissCount.incrementAndGet();
        KlineBackfillProvider provider = findBackfillProvider(request.seriesId());
        if (provider == null) {
            return cached;
        }
        backfillRequestCount.incrementAndGet();
        List<KlineBar> loaded = provider.load(request);
        if (loaded != null && !loaded.isEmpty()) {
            backfillBarsLoaded.addAndGet(loaded.size());
            appendBars(loaded);
        }
        return buffer.select(request);
    }

    @Override
    public KlineSubscription subscribe(KlineSubscriptionRequest request, KlineConsumer consumer) {
        Objects.requireNonNull(request, "request must not be null");
        Objects.requireNonNull(consumer, "consumer must not be null");
        try {
            if (request.replayBackfill() && request.backfillRequest() != null) {
                List<KlineBar> backfill = requestBars(request.backfillRequest().withClosedBarsOnly(request.closedBarsOnly()));
                consumer.onBackfill(request.seriesId(), backfill);
            }
        } catch (Exception error) {
            consumer.onError(request.seriesId(), error instanceof Exception ? (Exception) error : new RuntimeException(error));
        }
        SubscriptionImpl subscription = new SubscriptionImpl(
                "sub-" + subscriptionSequence.getAndIncrement() + "-" + UUID.randomUUID(),
                request.seriesId(),
                request.closedBarsOnly(),
                consumer);
        subscriptions.computeIfAbsent(request.seriesId(), id -> new CopyOnWriteArrayList<>()).add(subscription);
        return subscription;
    }

    public KlineFeedMetricsSnapshot metricsSnapshot(Duration staleAfter) {
        Objects.requireNonNull(staleAfter, "staleAfter must not be null");
        long generatedAt = clock.millis();
        List<KlineFeedProducerStatus> producerStatuses = new ArrayList<>();
        for (ProducerState state : producerStates.values()) {
            producerStatuses.add(state.snapshot(generatedAt, staleAfter.toMillis()));
        }
        producerStatuses.sort(Comparator.comparing(KlineFeedProducerStatus::producerId));
        long totalRequests = requestCount.get();
        long hits = cacheHitCount.get();
        long misses = cacheMissCount.get();
        long published = publishedBarCount.get();
        double hitRate = totalRequests == 0L ? 1.0d : (double) hits / (double) totalRequests;
        double avgLatency = published == 0L ? 0.0d : (double) totalFeedLatencyMillis.get() / (double) published;
        return new KlineFeedMetricsSnapshot(
                generatedAt,
                staleAfter.toMillis(),
                totalRequests,
                hits,
                misses,
                hitRate,
                backfillRequestCount.get(),
                backfillBarsLoaded.get(),
                published,
                avgLatency,
                maxFeedLatencyMillis.get(),
                lastFeedLatencyMillis.get(),
                buffers.size(),
                totalBufferedBars(),
                totalSubscriptions(),
                maxBarsPerSeries,
                producerStatuses);
    }

    public KlineFeedHealthReport healthReport(Duration staleAfter) {
        KlineFeedMetricsSnapshot snapshot = metricsSnapshot(staleAfter);
        List<String> alerts = new ArrayList<>();
        if (snapshot.producers().isEmpty()) {
            alerts.add("no producers registered");
        }
        int staleCount = 0;
        for (KlineFeedProducerStatus producer : snapshot.producers()) {
            if (producer.stale()) {
                staleCount++;
                alerts.add("producer " + producer.producerId() + " stale for " + producer.lastPublishAgeMillis() + " ms");
            }
        }
        String status = alerts.isEmpty()
                ? "OK"
                : staleCount == snapshot.producers().size() && !snapshot.producers().isEmpty() ? "CRITICAL" : "WARN";
        return new KlineFeedHealthReport(status, snapshot.generatedAtMillis(), snapshot, alerts);
    }

    public String prometheusMetrics(Duration staleAfter) {
        KlineFeedMetricsSnapshot snapshot = metricsSnapshot(staleAfter);
        StringBuilder text = new StringBuilder(1024);
        appendMetric(text, "xtrade_kline_cache_requests_total", snapshot.cacheRequests());
        appendMetric(text, "xtrade_kline_cache_hits_total", snapshot.cacheHits());
        appendMetric(text, "xtrade_kline_cache_misses_total", snapshot.cacheMisses());
        appendMetric(text, "xtrade_kline_cache_hit_rate", snapshot.cacheHitRate());
        appendMetric(text, "xtrade_kline_backfill_requests_total", snapshot.backfillRequests());
        appendMetric(text, "xtrade_kline_backfill_bars_total", snapshot.backfillBarsLoaded());
        appendMetric(text, "xtrade_kline_feed_latency_millis_avg", snapshot.averageFeedLatencyMillis());
        appendMetric(text, "xtrade_kline_feed_latency_millis_max", snapshot.maxFeedLatencyMillis());
        appendMetric(text, "xtrade_kline_feed_latency_millis_last", snapshot.lastFeedLatencyMillis());
        appendMetric(text, "xtrade_kline_buffered_bars", snapshot.bufferedBars());
        appendMetric(text, "xtrade_kline_active_subscriptions", snapshot.activeSubscriptions());
        appendMetric(text, "xtrade_kline_registered_series", snapshot.seriesCount());
        for (KlineFeedProducerStatus producer : snapshot.producers()) {
            String labels = "{producer_id=\"" + escapeLabel(producer.producerId()) + "\",producer_name=\""
                    + escapeLabel(producer.producerName()) + "\"}";
            appendMetric(text, "xtrade_kline_producer_connected" + labels, producer.connected() ? 1 : 0);
            appendMetric(text, "xtrade_kline_producer_stale" + labels, producer.stale() ? 1 : 0);
            appendMetric(text, "xtrade_kline_producer_last_publish_age_millis" + labels, producer.lastPublishAgeMillis());
            appendMetric(text, "xtrade_kline_producer_published_bars_total" + labels, producer.publishedBars());
        }
        return text.toString();
    }

    private void appendBars(Collection<KlineBar> bars) {
        for (KlineBar bar : bars) {
            SeriesBuffer buffer = bufferFor(bar.seriesId());
            buffer.upsert(bar);
            recordPublishedBar(bar);
            CopyOnWriteArrayList<SubscriptionImpl> listeners = subscriptions.get(bar.seriesId());
            if (listeners != null && !listeners.isEmpty()) {
                for (SubscriptionImpl listener : listeners) {
                    if (listener.accepts(bar)) {
                        listener.consumer.onLiveBar(bar);
                    }
                }
            }
        }
    }

    private SeriesBuffer bufferFor(KlineSeriesId seriesId) {
        return buffers.computeIfAbsent(seriesId, id -> new SeriesBuffer(maxBarsPerSeries));
    }

    private KlineBackfillProvider findBackfillProvider(KlineSeriesId seriesId) {
        for (ProducerHandleImpl producer : producers.values()) {
            if (producer.registration.supports(seriesId) && producer.registration.backfillProvider() != null) {
                return producer.registration.backfillProvider();
            }
        }
        return null;
    }

    private boolean isSatisfied(KlineBatchRequest request, List<KlineBar> selected, SeriesBuffer buffer) {
        if (request.limit() != null) {
            return selected.size() >= request.limit();
        }
        return buffer.covers(request, selected);
    }

    private void recordPublishedBar(KlineBar bar) {
        publishedBarCount.incrementAndGet();
        long latency = Math.max(0L, bar.ingestTimeMillis() - bar.eventTimeMillis());
        lastFeedLatencyMillis.set(latency);
        totalFeedLatencyMillis.addAndGet(latency);
        maxFeedLatencyMillis.accumulateAndGet(latency, Math::max);
        ProducerState state = producerStates.get(bar.producerId());
        if (state != null) {
            state.recordPublish(bar);
        }
    }

    private long totalSubscriptions() {
        long total = 0L;
        for (CopyOnWriteArrayList<SubscriptionImpl> value : subscriptions.values()) {
            total += value.size();
        }
        return total;
    }

    private long totalBufferedBars() {
        long total = 0L;
        for (SeriesBuffer buffer : buffers.values()) {
            total += buffer.size();
        }
        return total;
    }

    private static void appendMetric(StringBuilder text, String name, double value) {
        text.append(name).append(' ').append(value).append('\n');
    }

    private static void appendMetric(StringBuilder text, String name, long value) {
        text.append(name).append(' ').append(value).append('\n');
    }

    private static String escapeLabel(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private final class ProducerHandleImpl implements KlineProducerHandle {
        private final KlineProducerRegistration registration;

        private ProducerHandleImpl(KlineProducerRegistration registration) {
            this.registration = registration;
        }

        @Override
        public KlineProducerRegistration registration() {
            return registration;
        }

        @Override
        public void publish(KlineBar bar) {
            Objects.requireNonNull(bar, "bar must not be null");
            validateProducedSeries(bar.seriesId());
            appendBars(Collections.singletonList(bar));
        }

        @Override
        public void publishAll(Collection<KlineBar> bars) {
            Objects.requireNonNull(bars, "bars must not be null");
            for (KlineBar bar : bars) {
                Objects.requireNonNull(bar, "bars must not contain nulls");
                validateProducedSeries(bar.seriesId());
            }
            appendBars(bars);
        }

        private void validateProducedSeries(KlineSeriesId seriesId) {
            if (!registration.supports(seriesId)) {
                throw new IllegalArgumentException("producer " + registration.producerId() + " is not registered for " + seriesId);
            }
        }
    }

    private final class SubscriptionImpl implements KlineSubscription {
        private final String id;
        private final KlineSeriesId seriesId;
        private final boolean closedBarsOnly;
        private final KlineConsumer consumer;

        private SubscriptionImpl(String id, KlineSeriesId seriesId, boolean closedBarsOnly, KlineConsumer consumer) {
            this.id = id;
            this.seriesId = seriesId;
            this.closedBarsOnly = closedBarsOnly;
            this.consumer = consumer;
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public void close() {
            CopyOnWriteArrayList<SubscriptionImpl> listeners = subscriptions.get(seriesId);
            if (listeners != null) {
                listeners.remove(this);
            }
            consumer.onClosed(seriesId);
        }

        private boolean accepts(KlineBar bar) {
            return !closedBarsOnly || bar.closed();
        }
    }

    private static final class ProducerState {
        private final KlineProducerRegistration registration;
        private final long registeredAtMillis;
        private final AtomicLong lastPublishTimeMillis = new AtomicLong(-1L);
        private final AtomicLong publishedBars = new AtomicLong();
        private final AtomicBoolean seenPublish = new AtomicBoolean(false);

        private ProducerState(KlineProducerRegistration registration, long registeredAtMillis) {
            this.registration = registration;
            this.registeredAtMillis = registeredAtMillis;
        }

        private void recordPublish(KlineBar bar) {
            seenPublish.set(true);
            publishedBars.incrementAndGet();
            lastPublishTimeMillis.accumulateAndGet(bar.ingestTimeMillis(), Math::max);
        }

        private KlineFeedProducerStatus snapshot(long nowMillis, long staleAfterMillis) {
            long lastPublish = lastPublishTimeMillis.get();
            long age = seenPublish.get() ? Math.max(0L, nowMillis - lastPublish) : Math.max(0L, nowMillis - registeredAtMillis);
            boolean stale = age > staleAfterMillis;
            boolean connected = !stale;
            return new KlineFeedProducerStatus(
                    registration.producerId(),
                    registration.producerName(),
                    registration.publishedSeries().size(),
                    lastPublish,
                    age,
                    publishedBars.get(),
                    connected,
                    stale);
        }
    }

    private static final class SeriesBuffer {
        private static final Comparator<KlineBar> ORDER_BY_OPEN_TIME = Comparator.comparingLong(KlineBar::openTimeMillis);

        private final int maxBars;
        private final List<KlineBar> bars = new ArrayList<>();

        private SeriesBuffer(int maxBars) {
            this.maxBars = maxBars;
        }

        private synchronized void upsert(KlineBar bar) {
            int existingIndex = indexOf(bar.openTimeMillis());
            if (existingIndex >= 0) {
                KlineBar existing = bars.get(existingIndex);
                if (shouldReplace(existing, bar)) {
                    bars.set(existingIndex, bar);
                }
            } else {
                bars.add(bar);
                bars.sort(ORDER_BY_OPEN_TIME);
            }
            while (bars.size() > maxBars) {
                bars.remove(0);
            }
        }

        private synchronized List<KlineBar> select(KlineBatchRequest request) {
            List<KlineBar> selected = new ArrayList<>();
            for (KlineBar bar : bars) {
                if (request.closedBarsOnly() && !bar.closed()) {
                    continue;
                }
                if (request.startTimeMillisInclusive() != null && bar.openTimeMillis() < request.startTimeMillisInclusive()) {
                    continue;
                }
                if (request.endTimeMillisExclusive() != null && bar.openTimeMillis() >= request.endTimeMillisExclusive()) {
                    continue;
                }
                selected.add(bar);
            }
            if (request.limit() != null && selected.size() > request.limit()) {
                return new ArrayList<>(selected.subList(selected.size() - request.limit(), selected.size()));
            }
            return selected;
        }

        private synchronized boolean covers(KlineBatchRequest request, List<KlineBar> selected) {
            if (request.startTimeMillisInclusive() == null || request.endTimeMillisExclusive() == null) {
                return !selected.isEmpty();
            }
            if (selected.isEmpty()) {
                return false;
            }
            long expected = request.startTimeMillisInclusive();
            long step = request.seriesId().interval().toMillis();
            for (KlineBar bar : selected) {
                if (bar.openTimeMillis() != expected) {
                    return false;
                }
                expected += step;
            }
            return expected >= request.endTimeMillisExclusive();
        }

        private static boolean shouldReplace(KlineBar existing, KlineBar candidate) {
            if (candidate.closed() && !existing.closed()) {
                return true;
            }
            if (candidate.sequence() > existing.sequence()) {
                return true;
            }
            return candidate.eventTimeMillis() >= existing.eventTimeMillis();
        }

        private int indexOf(long openTimeMillis) {
            for (int i = 0; i < bars.size(); i++) {
                if (bars.get(i).openTimeMillis() == openTimeMillis) {
                    return i;
                }
            }
            return -1;
        }

        private synchronized int size() {
            return bars.size();
        }
    }
}
