package com.xtrade.kline;

import com.xtrade.PaperTradingEngine;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.math.BigDecimal;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DrawThruCachingKlineFeedTest {

    private static final KlineSeriesId BTC_1M = new KlineSeriesId("binance", "BTC/USD", KlineInterval.ONE_MINUTE);

    @Test
    void barSchemaCarriesCanonicalFieldsAndMetadata() {
        Map<String, String> metadata = new HashMap<>();
        metadata.put("exchangeSymbol", "BTCUSDT");
        metadata.put("firstTradeId", "1000");
        metadata.put("lastTradeId", "1012");
        metadata.put("timezone", "UTC");

        KlineBar bar = bar(0L, "100.0", "105.0", "99.0", "103.0", true, 7L, metadata);

        assertEquals(BTC_1M, bar.seriesId());
        assertEquals(0L, bar.openTimeMillis());
        assertEquals(60_000L, bar.closeTimeMillis());
        assertEquals(new BigDecimal("100.0"), bar.openPrice());
        assertEquals(new BigDecimal("105.0"), bar.highPrice());
        assertEquals(new BigDecimal("99.0"), bar.lowPrice());
        assertEquals(new BigDecimal("103.0"), bar.closePrice());
        assertEquals(new BigDecimal("12.5"), bar.baseVolume());
        assertEquals(new BigDecimal("1287.5"), bar.quoteVolume());
        assertEquals(42L, bar.tradeCount());
        assertEquals(new BigDecimal("7.0"), bar.takerBuyBaseVolume());
        assertEquals(new BigDecimal("721.0"), bar.takerBuyQuoteVolume());
        assertTrue(bar.closed());
        assertEquals(7L, bar.sequence());
        assertEquals(KlineSource.WEBSOCKET_MUX, bar.source());
        assertEquals("BTCUSDT", bar.metadata("exchangeSymbol"));
        assertEquals("1000", bar.metadata("firstTradeId"));
        assertEquals("1012", bar.metadata("lastTradeId"));
        assertEquals("UTC", bar.metadata("timezone"));
    }

    @Test
    void requestBarsDrawsThroughBackfillAndCachesResults() {
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(32);
        AtomicInteger backfillCalls = new AtomicInteger();
        feed.registerProducer(new KlineProducerRegistration(
                "binance-rest",
                "binance rest backfill",
                Collections.singleton(BTC_1M),
                request -> {
                    backfillCalls.incrementAndGet();
                    return Arrays.asList(
                            bar(0L, "100", "105", "99", "102", true, 1L),
                            bar(60_000L, "102", "106", "101", "105", true, 2L)
                    );
                }));

        List<KlineBar> first = feed.requestBars(KlineBatchRequest.between(BTC_1M, 0L, 120_000L));
        List<KlineBar> second = feed.requestBars(KlineBatchRequest.between(BTC_1M, 0L, 120_000L));

        assertEquals(1, backfillCalls.get());
        assertEquals(2, first.size());
        assertEquals(2, second.size());
        assertEquals(new BigDecimal("105"), second.get(1).closePrice());
    }

    @Test
    void healthAndMetricsExposeCacheHitRateLatencyAndProducerConnectivity() throws Exception {
        MutableClock clock = new MutableClock(Instant.parse("2026-04-17T09:25:00Z"));
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(32, clock);
        KlineProducerHandle producer = feed.registerProducer(new KlineProducerRegistration(
                "binance-live",
                "binance websocket",
                Collections.singleton(BTC_1M),
                request -> Collections.singletonList(bar(BTC_1M, 0L, "100", "101", "99", "100.5", true, 1L, 15L, Collections.singletonMap("exchangeSymbol", "BTCUSDT")))));

        feed.requestBars(KlineBatchRequest.latest(BTC_1M, 1));
        feed.requestBars(KlineBatchRequest.latest(BTC_1M, 1));
        producer.publish(bar(BTC_1M, 60_000L, "100.5", "102", "100", "101.75", true, 2L, 25L, Collections.singletonMap("exchangeSymbol", "BTCUSDT")));

        KlineFeedHealthReport health = feed.healthReport(Duration.ofSeconds(90));
        assertEquals("OK", health.status());
        assertTrue(health.healthy());
        assertEquals(2L, health.metrics().cacheRequests());
        assertEquals(1L, health.metrics().cacheHits());
        assertEquals(1L, health.metrics().cacheMisses());
        assertEquals(0.5d, health.metrics().cacheHitRate(), 1e-9d);
        assertEquals(2L, health.metrics().publishedBars());
        assertEquals(25L, health.metrics().maxFeedLatencyMillis());
        assertEquals(25L, health.metrics().lastFeedLatencyMillis());
        assertEquals(20.0d, health.metrics().averageFeedLatencyMillis(), 1e-9d);
        assertEquals(1, health.metrics().producers().size());
        assertTrue(health.metrics().producers().get(0).connected());
        assertFalse(health.metrics().producers().get(0).stale());

        String metrics = feed.prometheusMetrics(Duration.ofSeconds(90));
        assertTrue(metrics.contains("xtrade_kline_cache_hit_rate 0.5"));
        assertTrue(metrics.contains("xtrade_kline_feed_latency_millis_avg 20.0"));
        assertTrue(metrics.contains("xtrade_kline_producer_connected{producer_id=\"binance-live\""));

        clock.advanceMillis(Duration.ofMinutes(5).toMillis());
        KlineFeedHealthReport staleHealth = feed.healthReport(Duration.ofSeconds(90));
        assertEquals("CRITICAL", staleHealth.status());
        assertFalse(staleHealth.alerts().isEmpty());
        assertTrue(staleHealth.alerts().get(0).contains("stale"));
    }

    @Test
    void monitorServerPublishesHealthEndpointAndPrometheusMetrics() throws Exception {
        MutableClock clock = new MutableClock(Instant.parse("2026-04-17T09:25:00Z"));
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(32, clock);
        KlineProducerHandle producer = feed.registerProducer(new KlineProducerRegistration(
                "binance-live",
                "binance websocket",
                Collections.singleton(BTC_1M),
                null));
        producer.publish(bar(BTC_1M, 0L, "100", "101", "99", "100.5", true, 1L, 10L, Collections.singletonMap("exchangeSymbol", "BTCUSDT")));

        try (KlineFeedMonitorServer server = KlineFeedMonitorServer.start(feed, new InetSocketAddress("127.0.0.1", 0), Duration.ofSeconds(90))) {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest healthRequest = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/health")).GET().build();
            HttpRequest metricsRequest = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/metrics")).GET().build();

            HttpResponse<String> healthResponse = client.send(healthRequest, HttpResponse.BodyHandlers.ofString());
            HttpResponse<String> metricsResponse = client.send(metricsRequest, HttpResponse.BodyHandlers.ofString());

            assertEquals(200, healthResponse.statusCode());
            assertEquals(200, metricsResponse.statusCode());
            assertTrue(healthResponse.body().contains("\"status\": \"OK\""));
            assertTrue(healthResponse.body().contains("\"cacheHitRate\": 1.0"));
            assertTrue(metricsResponse.body().contains("xtrade_kline_cache_hit_rate 1.0"));
            assertTrue(metricsResponse.body().contains("xtrade_kline_producer_connected{producer_id=\"binance-live\""));
        }
    }

    @Test
    void subscribeReplaysBackfillThenStreamsForward() {
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(32);
        AtomicInteger backfillCalls = new AtomicInteger();
        KlineProducerHandle producer = feed.registerProducer(new KlineProducerRegistration(
                "binance-mux",
                "binance live mux",
                Collections.singleton(BTC_1M),
                request -> {
                    backfillCalls.incrementAndGet();
                    return Collections.singletonList(bar(0L, "100", "101", "99", "100.5", true, 1L));
                }));

        RecordingConsumer consumer = new RecordingConsumer();
        KlineSubscription subscription = feed.subscribe(
                KlineSubscriptionRequest.backfillThenLive(KlineBatchRequest.latest(BTC_1M, 1)),
                consumer);

        producer.publish(bar(60_000L, "100.5", "103", "100", "102.25", false, 2L));
        producer.publish(bar(60_000L, "100.5", "103", "100", "102.50", true, 3L));
        subscription.close();

        assertEquals(1, backfillCalls.get());
        assertEquals(1, consumer.backfill.size());
        assertEquals(new BigDecimal("100.5"), consumer.backfill.get(0).closePrice());
        assertEquals(1, consumer.live.size(), "closedBarsOnly should suppress the open update");
        assertEquals(new BigDecimal("102.50"), consumer.live.get(0).closePrice());
        assertTrue(consumer.closed);
    }

    @Test
    void paperTradingAdapterConsumesHistoricalAndLiveBars() {
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(32);
        KlineProducerHandle producer = feed.registerProducer(new KlineProducerRegistration(
                "binance-paper",
                "paper engine feed",
                Collections.singleton(BTC_1M),
                request -> Collections.singletonList(bar(0L, "100", "101", "99", "100.5", true, 1L))));

        PaperTradingEngine engine = new PaperTradingEngine(10_000.0, true);
        PaperTradingEngineKlineAdapter adapter = new PaperTradingEngineKlineAdapter(engine);
        KlineSubscription subscription = feed.subscribe(
                KlineSubscriptionRequest.backfillThenLive(KlineBatchRequest.latest(BTC_1M, 1)),
                adapter);

        assertEquals(100.5, engine.getMarketPrice("BTC/USD"), 1e-9);

        producer.publish(bar(60_000L, "100.5", "102", "100", "101.75", true, 2L));
        assertEquals(101.75, engine.getMarketPrice("BTC/USD"), 1e-9);

        subscription.close();
    }

    @Test
    void producerCannotPublishSeriesItDidNotRegister() {
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(8);
        KlineProducerHandle producer = feed.registerProducer(new KlineProducerRegistration(
                "binance-btc-only",
                "binance btc only",
                Collections.singleton(BTC_1M),
                null));
        KlineSeriesId ethSeries = new KlineSeriesId("binance", "ETH/USD", KlineInterval.ONE_MINUTE);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> producer.publish(bar(ethSeries, 0L, "10", "11", "9", "10.5", true, 1L, Collections.emptyMap())));
        assertTrue(error.getMessage().contains("not registered"));
    }

    @Test
    void intervalRegistryParsesWireNamesAndAlignsOpenTimes() {
        assertEquals(KlineInterval.ONE_MINUTE, KlineInterval.parse("1m"));
        assertEquals(KlineInterval.ONE_MONTH, KlineInterval.parse("1M"));
        assertEquals(120_000L, KlineInterval.ONE_MINUTE.alignOpenTime(120_123L));
        assertEquals(180_000L, KlineInterval.ONE_MINUTE.closeTimeExclusive(120_000L));
        assertFalse(KlineInterval.ONE_MINUTE.equals(KlineInterval.FIVE_MINUTES));
    }

    private static KlineBar bar(long openTimeMillis,
                                String open,
                                String high,
                                String low,
                                String close,
                                boolean closed,
                                long sequence) {
        return bar(BTC_1M, openTimeMillis, open, high, low, close, closed, sequence, Collections.singletonMap("exchangeSymbol", "BTCUSDT"));
    }

    private static KlineBar bar(long openTimeMillis,
                                String open,
                                String high,
                                String low,
                                String close,
                                boolean closed,
                                long sequence,
                                Map<String, String> metadata) {
        return bar(BTC_1M, openTimeMillis, open, high, low, close, closed, sequence, metadata);
    }

    private static KlineBar bar(KlineSeriesId seriesId,
                                long openTimeMillis,
                                String open,
                                String high,
                                String low,
                                String close,
                                boolean closed,
                                long sequence,
                                Map<String, String> metadata) {
        return bar(seriesId, openTimeMillis, open, high, low, close, closed, sequence,
                seriesId.interval().toMillis() - 1L, metadata);
    }

    private static KlineBar bar(KlineSeriesId seriesId,
                                long openTimeMillis,
                                String open,
                                String high,
                                String low,
                                String close,
                                boolean closed,
                                long sequence,
                                long latencyMillis,
                                Map<String, String> metadata) {
        long eventTimeMillis = openTimeMillis + seriesId.interval().toMillis() - 1L;
        return new KlineBar(
                seriesId,
                openTimeMillis,
                openTimeMillis + seriesId.interval().toMillis(),
                eventTimeMillis,
                eventTimeMillis + latencyMillis,
                new BigDecimal(open),
                new BigDecimal(high),
                new BigDecimal(low),
                new BigDecimal(close),
                new BigDecimal("12.5"),
                new BigDecimal("1287.5"),
                42L,
                new BigDecimal("7.0"),
                new BigDecimal("721.0"),
                closed,
                sequence,
                "test-producer",
                KlineSource.WEBSOCKET_MUX,
                metadata);
    }

    private static final class RecordingConsumer implements KlineConsumer {
        private final List<KlineBar> backfill = new ArrayList<>();
        private final List<KlineBar> live = new ArrayList<>();
        private boolean closed;

        @Override
        public void onBackfill(KlineSeriesId seriesId, List<KlineBar> bars) {
            backfill.addAll(bars);
        }

        @Override
        public void onLiveBar(KlineBar bar) {
            live.add(bar);
        }

        @Override
        public void onClosed(KlineSeriesId seriesId) {
            closed = true;
        }
    }

    private static final class MutableClock extends Clock {
        private Instant instant;

        private MutableClock(Instant instant) {
            this.instant = instant;
        }

        @Override
        public ZoneId getZone() {
            return ZoneOffset.UTC;
        }

        @Override
        public Clock withZone(ZoneId zone) {
            return this;
        }

        @Override
        public Instant instant() {
            return instant;
        }

        private void advanceMillis(long millis) {
            instant = instant.plusMillis(millis);
        }
    }
}
