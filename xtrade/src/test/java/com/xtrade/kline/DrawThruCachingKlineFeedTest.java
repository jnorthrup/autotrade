package com.xtrade.kline;

import com.xtrade.PaperTradingEngine;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
        return new KlineBar(
                seriesId,
                openTimeMillis,
                openTimeMillis + seriesId.interval().toMillis(),
                openTimeMillis + seriesId.interval().toMillis() - 1L,
                openTimeMillis + seriesId.interval().toMillis() - 1L,
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
}
