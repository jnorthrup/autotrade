package com.xtrade.kline.binance;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import com.xtrade.kline.DrawThruCachingKlineFeed;
import com.xtrade.kline.KlineBar;
import com.xtrade.kline.KlineBatchRequest;
import com.xtrade.kline.KlineConsumer;
import com.xtrade.kline.KlineInterval;
import com.xtrade.kline.KlineSeriesId;
import com.xtrade.kline.KlineSubscription;
import com.xtrade.kline.KlineSubscriptionRequest;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BinanceKlineIntegrationTest {

    @Test
    void archiveFetchMergesMonthlyAndDailyArchivesIntoCanonicalCsvAndFeed() throws Exception {
        Instant fixedInstant = LocalDate.of(2024, 1, 2).atStartOfDay().toInstant(ZoneOffset.UTC);
        MutableClock clock = new MutableClock(fixedInstant);
        byte[] monthlyZip = zip("BTCUSDT-1m-2024-01.csv",
                "Open_time,Open,High,Low,Close,Volume,Close_time,Quote_asset_volume,Number_of_trades,Taker_buy_base_asset_volume,Taker_buy_quote_asset_volume,Ignore\n"
                        + "0,100,101,99,100.5,1.5,60000,150.75,10,0.9,90.45,0\n"
                        + "60000,100.5,102,100,101.5,2.0,120000,203,12,1.1,111.65,0\n");
        byte[] dailyZip = zip("BTCUSDT-1m-2024-01-02.csv",
                "0,100,101,99,100.5,1.5,60000,150.75,10,0.9,90.45,0\n"
                        + "120000,101.5,103,101,102.25,2.5,180000,255.625,14,1.3,132.925,0\n"
                        + "broken,row\n");

        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip", exchange -> reply(exchange, 200, monthlyZip, "application/zip"));
        server.createContext("/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-02.zip", exchange -> reply(exchange, 200, dailyZip, "application/zip"));
        server.createContext("/", exchange -> reply(exchange, 404, new byte[0], "text/plain"));
        server.start();
        try {
            DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(64);
            Path temp = Files.createTempDirectory("archive-fetch");
            BinanceArchiveFetchService service = new BinanceArchiveFetchService(
                    new JdkBinanceHttpClient(),
                    new URI("http://127.0.0.1:" + server.getAddress().getPort()),
                    clock,
                    2024,
                    feed);
            BinanceArchiveFetchService.FetchSpec spec = new BinanceArchiveFetchService.FetchSpec(
                    "BTC", "USDT", KlineInterval.ONE_MINUTE,
                    temp.resolve("cache"),
                    temp.resolve("import"));

            BinanceArchiveFetchService.FetchResult result = service.fetch(spec);
            List<String> lines = Files.readAllLines(result.csvPath(), StandardCharsets.UTF_8);
            List<KlineBar> bars = feed.requestBars(KlineBatchRequest.between(spec.seriesId(), 0L, 180_000L));

            assertEquals(4, lines.size());
            assertEquals(BinanceKlineRecord.CSV_HEADER, lines.get(0));
            assertTrue(lines.get(1).startsWith("0,100,101,99,100.5"));
            assertTrue(lines.get(3).startsWith("120000,101.5,103,101,102.25"));
            assertEquals(3, result.rowCount());
            assertEquals(3, bars.size());
            assertEquals("102.25", bars.get(2).closePrice().toPlainString());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void incrementalFetcherSupportsMultiSymbolFetchAndRateLimiting() throws Exception {
        MutableClock clock = new MutableClock(Instant.parse("2024-01-02T00:00:00Z"));
        List<Long> sleeps = new ArrayList<>();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        Map<String, AtomicInteger> calls = new LinkedHashMap<>();
        server.createContext("/api/v3/klines", exchange -> {
            String query = exchange.getRequestURI().getQuery();
            String symbol = param(query, "symbol");
            calls.computeIfAbsent(symbol, ignored -> new AtomicInteger()).incrementAndGet();
            String body;
            if ("BTCUSDT".equals(symbol)) {
                assertEquals("60000", param(query, "startTime"));
                body = "[[60000,\"100.5\",\"102\",\"100\",\"101.5\",\"2\",120000,\"203\",12,\"1.1\",\"111.65\",\"0\"],[120000,\"101.5\",\"103\",\"101\",\"102.5\",\"2.2\",180000,\"225.5\",13,\"1.2\",\"123.4\",\"0\"]]";
            } else {
                assertEquals("60000", param(query, "startTime"));
                body = "[[60000,\"10.5\",\"12\",\"10\",\"11.5\",\"5\",120000,\"57.5\",22,\"2.1\",\"24.15\",\"0\"],[120000,\"11.5\",\"13\",\"11\",\"12.5\",\"6\",180000,\"75\",25,\"2.4\",\"30\",\"0\"]]";
            }
            reply(exchange, 200, body.getBytes(StandardCharsets.UTF_8), "application/json");
        });
        server.start();
        try {
            Path root = Files.createTempDirectory("incremental-fetch");
            Path btcTarget = root.resolve("BTC");
            Path ethTarget = root.resolve("ETH");
            Files.createDirectories(btcTarget);
            Files.createDirectories(ethTarget);
            Files.writeString(btcTarget.resolve("final-BTC-USDT-1m.csv"), BinanceKlineRecord.CSV_HEADER + "\n0,100,101,99,100.5,1.5,60000,150.75,10,0.9,90.45,0\n", StandardCharsets.UTF_8);
            Files.writeString(ethTarget.resolve("final-ETH-USDT-1m.csv"), BinanceKlineRecord.CSV_HEADER + "\n0,10,11,9,10.5,4.5,60000,47.25,20,1.9,19.95,0\n", StandardCharsets.UTF_8);

            DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(64);
            BinanceRateLimiter limiter = new BinanceRateLimiter(500L, clock, millis -> {
                sleeps.add(millis);
                clock.advanceMillis(millis);
            });
            BinanceIncrementalFetchService service = new BinanceIncrementalFetchService(
                    new JdkBinanceHttpClient(),
                    new URI("http://127.0.0.1:" + server.getAddress().getPort()),
                    limiter,
                    feed);

            Map<KlineSeriesId, Integer> counts = service.fetchAll(Arrays.asList(
                    new BinanceIncrementalFetchService.FetchSpec("BTC", "USDT", KlineInterval.ONE_MINUTE, btcTarget),
                    new BinanceIncrementalFetchService.FetchSpec("ETH", "USDT", KlineInterval.ONE_MINUTE, ethTarget)));

            assertEquals(2, counts.get(new KlineSeriesId("binance", "BTC/USDT", KlineInterval.ONE_MINUTE)));
            assertEquals(2, counts.get(new KlineSeriesId("binance", "ETH/USDT", KlineInterval.ONE_MINUTE)));
            assertEquals(List.of(500L), sleeps);
            assertEquals(1, calls.get("BTCUSDT").get());
            assertEquals(1, calls.get("ETHUSDT").get());
            assertTrue(Files.readString(btcTarget.resolve("final-BTC-USDT-1m.csv")).contains("120000,101.5,103,101,102.5"));
            assertTrue(Files.readString(ethTarget.resolve("final-ETH-USDT-1m.csv")).contains("120000,11.5,13,11,12.5"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void muxerPublishesOpenAndClosedEventsIntoUnifiedFeed() throws Exception {
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(64);
        KlineSeriesId seriesId = new KlineSeriesId("binance", "BTC/USDT", KlineInterval.ONE_MINUTE);
        BinanceKlineMuxer muxer = new BinanceKlineMuxer(feed, "binance-mux-test", List.of(seriesId));
        RecordingConsumer consumer = new RecordingConsumer();
        KlineSubscription subscription = feed.subscribe(
                KlineSubscriptionRequest.backfillThenLive(KlineBatchRequest.latest(seriesId, 5)),
                consumer);

        muxer.publishEventJson("{\"e\":\"kline\",\"E\":1704153600100,\"s\":\"BTCUSDT\",\"k\":{\"t\":1704153600000,\"T\":1704153659999,\"s\":\"BTCUSDT\",\"i\":\"1m\",\"f\":1,\"L\":10,\"o\":\"100\",\"c\":\"101\",\"h\":\"102\",\"l\":\"99\",\"v\":\"3.2\",\"n\":10,\"x\":false,\"q\":\"323.2\",\"V\":\"1.1\",\"Q\":\"111.1\"}}");
        muxer.publishEventJson("{\"e\":\"kline\",\"E\":1704153660000,\"s\":\"BTCUSDT\",\"k\":{\"t\":1704153600000,\"T\":1704153659999,\"s\":\"BTCUSDT\",\"i\":\"1m\",\"f\":1,\"L\":12,\"o\":\"100\",\"c\":\"101.5\",\"h\":\"103\",\"l\":\"99\",\"v\":\"3.9\",\"n\":12,\"x\":true,\"q\":\"395.85\",\"V\":\"1.3\",\"Q\":\"131.95\"}}");
        subscription.close();

        List<KlineBar> bars = feed.requestBars(KlineBatchRequest.latest(seriesId, 1));
        assertEquals(1, consumer.live.size());
        assertEquals("101.5", consumer.live.get(0).closePrice().toPlainString());
        assertEquals("101.5", bars.get(0).closePrice().toPlainString());
    }

    private static void reply(HttpExchange exchange, int status, byte[] body, String contentType) throws IOException {
        exchange.getResponseHeaders().add("Content-Type", contentType);
        exchange.sendResponseHeaders(status, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        }
    }

    private static byte[] zip(String fileName, String content) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (ZipOutputStream zip = new ZipOutputStream(out, StandardCharsets.UTF_8)) {
            zip.putNextEntry(new ZipEntry(fileName));
            zip.write(content.getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }
        return out.toByteArray();
    }

    private static String param(String query, String key) {
        for (String part : query.split("&")) {
            String[] pieces = part.split("=", 2);
            if (pieces[0].equals(key)) {
                return pieces.length > 1 ? pieces[1] : "";
            }
        }
        return "";
    }

    private static final class RecordingConsumer implements KlineConsumer {
        private final List<KlineBar> live = new ArrayList<>();

        @Override
        public void onBackfill(KlineSeriesId seriesId, List<KlineBar> bars) {
        }

        @Override
        public void onLiveBar(KlineBar bar) {
            live.add(bar);
        }

        @Override
        public void onError(KlineSeriesId seriesId, Exception error) {
            throw new RuntimeException(error);
        }

        @Override
        public void onClosed(KlineSeriesId seriesId) {
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
