package com.xtrade.kline.binance;

import com.xtrade.kline.DrawThruKlineFeed;
import com.xtrade.kline.KlineBar;
import com.xtrade.kline.KlineInterval;
import com.xtrade.kline.KlineProducerHandle;
import com.xtrade.kline.KlineProducerRegistration;
import com.xtrade.kline.KlineSeriesId;
import com.xtrade.kline.KlineSource;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class BinanceIncrementalFetchService {
    public static final class FetchSpec {
        private final String tradeAsset;
        private final String counterAsset;
        private final KlineInterval interval;
        private final Path targetDir;

        public FetchSpec(String tradeAsset, String counterAsset, KlineInterval interval, Path targetDir) {
            this.tradeAsset = tradeAsset.trim().toUpperCase();
            this.counterAsset = counterAsset.trim().toUpperCase();
            this.interval = interval;
            this.targetDir = targetDir;
        }

        public String exchangeSymbol() {
            return tradeAsset + counterAsset;
        }

        public KlineInterval interval() {
            return interval;
        }

        public Path finalCsv() {
            return targetDir.resolve("final-" + tradeAsset + "-" + counterAsset + "-" + interval.wireName() + ".csv");
        }

        public KlineSeriesId seriesId() {
            return new KlineSeriesId("binance", tradeAsset + "/" + counterAsset, interval);
        }
    }

    private final BinanceHttpClient httpClient;
    private final URI apiBaseUrl;
    private final BinanceRateLimiter rateLimiter;
    private final DrawThruKlineFeed feed;
    private final Map<KlineSeriesId, KlineProducerHandle> producers = new LinkedHashMap<>();

    public BinanceIncrementalFetchService(BinanceHttpClient httpClient,
                                          URI apiBaseUrl,
                                          BinanceRateLimiter rateLimiter,
                                          DrawThruKlineFeed feed) {
        this.httpClient = httpClient;
        this.apiBaseUrl = apiBaseUrl;
        this.rateLimiter = rateLimiter;
        this.feed = feed;
    }

    public Map<KlineSeriesId, Integer> fetchAll(Collection<FetchSpec> specs) throws IOException, InterruptedException {
        Map<KlineSeriesId, Integer> counts = new LinkedHashMap<>();
        for (FetchSpec spec : specs) {
            counts.put(spec.seriesId(), append(spec));
        }
        return counts;
    }

    public int append(FetchSpec spec) throws IOException, InterruptedException {
        Map<Long, BinanceKlineRecord> rows = BinanceKlineFileStore.read(spec.finalCsv());
        long since = BinanceKlineFileStore.lastCloseTime(spec.finalCsv());
        int appended = 0;
        while (true) {
            rateLimiter.acquire();
            URI uri = apiBaseUrl.resolve("/api/v3/klines?symbol=" + URLEncoder.encode(spec.exchangeSymbol(), StandardCharsets.UTF_8)
                    + "&interval=" + URLEncoder.encode(spec.interval().wireName(), StandardCharsets.UTF_8)
                    + (since >= 0 ? "&startTime=" + since : "")
                    + "&limit=1000");
            List<BinanceKlineRecord> batch = BinanceKlineRecord.parseRestPayload(httpClient.getString(uri));
            if (batch.isEmpty()) {
                break;
            }
            long previousSize = rows.size();
            for (BinanceKlineRecord record : batch) {
                rows.put(record.openTime(), record);
            }
            appended += rows.size() - previousSize;
            since = batch.get(batch.size() - 1).closeTime();
            if (batch.size() < 1000) {
                break;
            }
        }
        BinanceKlineFileStore.write(spec.finalCsv(), rows);
        publishRows(spec, rows.values());
        return appended;
    }

    private void publishRows(FetchSpec spec, Collection<BinanceKlineRecord> rows) {
        if (feed == null || rows.isEmpty()) {
            return;
        }
        KlineProducerHandle producer = producers.computeIfAbsent(spec.seriesId(), id -> feed.registerProducer(new KlineProducerRegistration(
                "binance-rest-append-" + spec.exchangeSymbol() + "-" + spec.interval().wireName(),
                "binance append producer " + spec.exchangeSymbol(),
                Set.of(spec.seriesId()),
                null)));
        List<KlineBar> bars = new ArrayList<>();
        long sequence = 1L;
        for (BinanceKlineRecord row : rows) {
            bars.add(row.toBar(spec.seriesId(), producer.registration().producerId(), KlineSource.REST_BACKFILL, sequence++, true));
        }
        producer.publishAll(bars);
    }
}
