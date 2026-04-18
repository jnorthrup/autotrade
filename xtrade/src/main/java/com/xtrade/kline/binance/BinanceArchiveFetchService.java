package com.xtrade.kline.binance;

import com.xtrade.kline.DrawThruKlineFeed;
import com.xtrade.kline.KlineBar;
import com.xtrade.kline.KlineBatchRequest;
import com.xtrade.kline.KlineInterval;
import com.xtrade.kline.KlineProducerHandle;
import com.xtrade.kline.KlineProducerRegistration;
import com.xtrade.kline.KlineSeriesId;
import com.xtrade.kline.KlineSource;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.LocalDate;
import java.time.YearMonth;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class BinanceArchiveFetchService {
    public static final class FetchSpec {
        private final String tradeAsset;
        private final String counterAsset;
        private final KlineInterval interval;
        private final Path cacheDir;
        private final Path targetDir;

        public FetchSpec(String tradeAsset, String counterAsset, KlineInterval interval, Path cacheDir, Path targetDir) {
            this.tradeAsset = tradeAsset.trim().toUpperCase();
            this.counterAsset = counterAsset.trim().toUpperCase();
            this.interval = interval;
            this.cacheDir = cacheDir;
            this.targetDir = targetDir;
        }

        public String tradeAsset() {
            return tradeAsset;
        }

        public String counterAsset() {
            return counterAsset;
        }

        public KlineInterval interval() {
            return interval;
        }

        public Path cacheDir() {
            return cacheDir;
        }

        public Path targetDir() {
            return targetDir;
        }

        public String exchangeSymbol() {
            return tradeAsset + counterAsset;
        }

        public KlineSeriesId seriesId() {
            return new KlineSeriesId("binance", tradeAsset + "/" + counterAsset, interval);
        }

        public Path finalCsv() {
            return targetDir.resolve("final-" + tradeAsset + "-" + counterAsset + "-" + interval.wireName() + ".csv");
        }
    }

    public static final class FetchResult {
        private final Path csvPath;
        private final int rowCount;

        FetchResult(Path csvPath, int rowCount) {
            this.csvPath = csvPath;
            this.rowCount = rowCount;
        }

        public Path csvPath() {
            return csvPath;
        }

        public int rowCount() {
            return rowCount;
        }
    }

    private final BinanceHttpClient httpClient;
    private final URI visionBaseUrl;
    private final Clock clock;
    private final int startYear;
    private final DrawThruKlineFeed feed;
    private final Map<KlineSeriesId, KlineProducerHandle> producers = new LinkedHashMap<>();

    public BinanceArchiveFetchService(BinanceHttpClient httpClient,
                                      URI visionBaseUrl,
                                      Clock clock,
                                      int startYear,
                                      DrawThruKlineFeed feed) {
        this.httpClient = httpClient;
        this.visionBaseUrl = visionBaseUrl;
        this.clock = clock;
        this.startYear = startYear;
        this.feed = feed;
    }

    public List<FetchResult> fetchAll(Collection<FetchSpec> specs) throws IOException, InterruptedException {
        List<FetchResult> results = new ArrayList<>();
        for (FetchSpec spec : specs) {
            results.add(fetch(spec));
        }
        return results;
    }

    public FetchResult fetch(FetchSpec spec) throws IOException, InterruptedException {
        Files.createDirectories(spec.cacheDir());
        Files.createDirectories(spec.targetDir());
        Map<Long, BinanceKlineRecord> rows = new LinkedHashMap<>();
        for (YearMonth month : monthRange()) {
            URI monthlyUri = visionBaseUrl.resolve(monthlyPath(spec, month));
            Path cached = spec.cacheDir().resolve(filename(monthlyUri));
            loadArchive(monthlyUri, cached, rows);
        }
        LocalDate now = LocalDate.now(clock.withZone(ZoneOffset.UTC));
        YearMonth currentMonth = YearMonth.from(now);
        for (int day = 1; day <= now.getDayOfMonth(); day++) {
            URI dailyUri = visionBaseUrl.resolve(dailyPath(spec, currentMonth.atDay(day)));
            Path cached = spec.cacheDir().resolve(filename(dailyUri));
            loadArchive(dailyUri, cached, rows);
        }
        BinanceKlineFileStore.write(spec.finalCsv(), rows);
        publishRows(spec, rows.values(), KlineSource.REST_BACKFILL, true);
        return new FetchResult(spec.finalCsv(), rows.size());
    }

    private void loadArchive(URI uri, Path cached, Map<Long, BinanceKlineRecord> rows) throws IOException, InterruptedException {
        byte[] bytes;
        if (Files.exists(cached)) {
            bytes = Files.readAllBytes(cached);
        } else {
            try {
                bytes = httpClient.getBytes(uri);
            } catch (IOException notFound) {
                return;
            }
            Files.write(cached, bytes);
        }
        readZipRows(new ByteArrayInputStream(bytes), rows);
    }

    private void readZipRows(InputStream inputStream, Map<Long, BinanceKlineRecord> rows) throws IOException {
        try (ZipInputStream zip = new ZipInputStream(inputStream, StandardCharsets.UTF_8)) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                if (!entry.getName().endsWith(".csv")) {
                    continue;
                }
                String text = new String(zip.readAllBytes(), StandardCharsets.UTF_8);
                for (String line : text.split("\\R")) {
                    BinanceKlineRecord record = BinanceKlineRecord.parseCsv(line);
                    if (record != null) {
                        rows.put(record.openTime(), record);
                    }
                }
            }
        }
    }

    private void publishRows(FetchSpec spec, Collection<BinanceKlineRecord> rows, KlineSource source, boolean closed) throws IOException {
        if (feed == null || rows.isEmpty()) {
            return;
        }
        KlineProducerHandle producer = producers.computeIfAbsent(spec.seriesId(), id -> feed.registerProducer(new KlineProducerRegistration(
                "binance-rest-" + spec.exchangeSymbol() + "-" + spec.interval().wireName(),
                "binance archive/rest producer " + spec.exchangeSymbol(),
                Set.of(spec.seriesId()),
                request -> loadBackfill(request, spec))));
        List<KlineBar> bars = new ArrayList<>();
        long sequence = 1L;
        for (BinanceKlineRecord row : rows) {
            bars.add(row.toBar(spec.seriesId(), producer.registration().producerId(), source, sequence++, closed));
        }
        producer.publishAll(bars);
    }

    private List<KlineBar> loadBackfill(KlineBatchRequest request, FetchSpec spec) {
        try {
            Map<Long, BinanceKlineRecord> rows = BinanceKlineFileStore.read(spec.finalCsv());
            List<KlineBar> result = new ArrayList<>();
            long sequence = 1L;
            for (BinanceKlineRecord row : rows.values()) {
                if (request.startTimeMillisInclusive() != null && row.openTime() < request.startTimeMillisInclusive()) {
                    continue;
                }
                if (request.endTimeMillisExclusive() != null && row.openTime() >= request.endTimeMillisExclusive()) {
                    continue;
                }
                result.add(row.toBar(spec.seriesId(), "binance-rest-backfill", KlineSource.REST_BACKFILL, sequence++, true));
            }
            if (request.limit() != null && result.size() > request.limit()) {
                return new ArrayList<>(result.subList(result.size() - request.limit(), result.size()));
            }
            return result;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private List<YearMonth> monthRange() {
        Set<YearMonth> months = new LinkedHashSet<>();
        YearMonth current = YearMonth.now(clock.withZone(ZoneOffset.UTC));
        for (int year = startYear; year <= current.getYear(); year++) {
            int endMonth = year == current.getYear() ? current.getMonthValue() : 12;
            for (int month = 1; month <= endMonth; month++) {
                months.add(YearMonth.of(year, month));
            }
        }
        return new ArrayList<>(months);
    }

    private static String monthlyPath(FetchSpec spec, YearMonth month) {
        return "/data/spot/monthly/klines/" + spec.exchangeSymbol() + "/" + spec.interval().wireName() + "/"
                + spec.exchangeSymbol() + "-" + spec.interval().wireName() + "-" + month + ".zip";
    }

    private static String dailyPath(FetchSpec spec, LocalDate day) {
        return "/data/spot/daily/klines/" + spec.exchangeSymbol() + "/" + spec.interval().wireName() + "/"
                + spec.exchangeSymbol() + "-" + spec.interval().wireName() + "-" + day + ".zip";
    }

    private static String filename(URI uri) {
        String path = uri.getPath();
        return path.substring(path.lastIndexOf('/') + 1);
    }
}
