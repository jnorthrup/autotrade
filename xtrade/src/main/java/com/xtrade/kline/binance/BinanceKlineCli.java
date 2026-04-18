package com.xtrade.kline.binance;

import com.xtrade.kline.DrawThruCachingKlineFeed;
import com.xtrade.kline.KlineInterval;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class BinanceKlineCli {
    private static final String SYMBOL_CATALOG_RESOURCE = "binance-spot-symbols.txt";

    private BinanceKlineCli() {
    }

    public static void main(String[] args) throws Exception {
        run(args);
    }

    public static int run(String[] args) throws Exception {
        if (args.length == 0 || "--help".equals(args[0]) || "help".equals(args[0])) {
            printHelp();
            return 0;
        }
        BinanceKlineConfig config = BinanceKlineConfig.load();
        switch (args[0]) {
            case "fetch":
                return runFetch(Arrays.copyOfRange(args, 1, args.length), config);
            case "day":
                return runDay(Arrays.copyOfRange(args, 1, args.length), config);
            case "meta":
                return runMeta(Arrays.copyOfRange(args, 1, args.length));
            default:
                throw new IllegalArgumentException("Unknown command: " + args[0]);
        }
    }

    private static int runFetch(String[] args, BinanceKlineConfig config) throws Exception {
        KlineInterval interval = parseInterval(args, KlineInterval.ONE_MINUTE);
        List<String> pairs = positionalPairs(args);
        Path cacheOverride = parsePathOption(args, "--cache-dir");
        Path targetOverride = parsePathOption(args, "--target-dir");
        if (pairs.isEmpty()) {
            throw new IllegalArgumentException("fetch requires at least one TRADE/COUNTER pair or TRADE COUNTER args");
        }
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(8192);
        BinanceArchiveFetchService service = new BinanceArchiveFetchService(new JdkBinanceHttpClient(), config.visionBaseUrl(), config.clock(), config.startYear(), feed);
        List<BinanceArchiveFetchService.FetchSpec> specs = new ArrayList<>();
        for (String pair : pairs) {
            String[] parts = pair.split("/");
            Path cacheDir = cacheOverride != null ? cacheOverride : config.defaultCacheDir(parts[0], parts[1], interval.wireName());
            Path targetDir = targetOverride != null ? targetOverride : config.defaultTargetDir(parts[0], parts[1], interval.wireName());
            specs.add(new BinanceArchiveFetchService.FetchSpec(parts[0], parts[1], interval, cacheDir, targetDir));
        }
        for (BinanceArchiveFetchService.FetchResult result : service.fetchAll(specs)) {
            System.out.println(result.csvPath() + " rows=" + result.rowCount());
        }
        return 0;
    }

    private static int runDay(String[] args, BinanceKlineConfig config) throws Exception {
        KlineInterval interval = parseInterval(args, KlineInterval.ONE_MINUTE);
        List<String> pairs = positionalPairs(args);
        Path targetOverride = parsePathOption(args, "--target-dir");
        if (pairs.isEmpty()) {
            throw new IllegalArgumentException("day requires at least one TRADE/COUNTER pair or TRADE COUNTER args");
        }
        DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(8192);
        BinanceIncrementalFetchService service = new BinanceIncrementalFetchService(
                new JdkBinanceHttpClient(),
                config.apiBaseUrl(),
                new BinanceRateLimiter(config.rateLimitMillis()),
                feed);
        List<BinanceIncrementalFetchService.FetchSpec> specs = new ArrayList<>();
        for (String pair : pairs) {
            String[] parts = pair.split("/");
            Path targetDir = targetOverride != null ? targetOverride : config.defaultTargetDir(parts[0], parts[1], interval.wireName());
            specs.add(new BinanceIncrementalFetchService.FetchSpec(parts[0], parts[1], interval, targetDir));
        }
        Map<com.xtrade.kline.KlineSeriesId, Integer> counts = service.fetchAll(specs);
        for (Map.Entry<com.xtrade.kline.KlineSeriesId, Integer> entry : counts.entrySet()) {
            System.out.println(entry.getKey() + " appended=" + entry.getValue());
        }
        return 0;
    }

    private static int runMeta(String[] args) throws IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("meta requires one or more base assets");
        }
        Map<String, Set<String>> suffixesByBase = loadSuffixes();
        for (String base : args) {
            Set<String> suffixes = suffixesByBase.getOrDefault(base.toUpperCase(), Set.of());
            if (suffixes.isEmpty()) {
                continue;
            }
            String joined = String.join(" ", suffixes);
            System.out.println("for i in " + joined + "; do binance-kline fetch " + base.toUpperCase() + " $i ; done");
        }
        return 0;
    }

    private static Map<String, Set<String>> loadSuffixes() throws IOException {
        try (InputStream in = BinanceKlineCli.class.getClassLoader().getResourceAsStream(SYMBOL_CATALOG_RESOURCE)) {
            if (in == null) {
                throw new IOException("Missing classpath resource: " + SYMBOL_CATALOG_RESOURCE);
            }
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return parseSuffixes(text);
        }
    }

    static Map<String, Set<String>> parseSuffixes(String text) {
        Pattern token = Pattern.compile("^([A-Z0-9]+)$", Pattern.MULTILINE);
        Matcher matcher = token.matcher(text);
        Map<String, Set<String>> result = new LinkedHashMap<>();
        while (matcher.find()) {
            String pair = matcher.group(1);
            for (String quote : List.of("USDT", "BUSD", "USDC", "BTC", "ETH", "BNB", "EUR", "GBP", "TRY", "BRL", "AUD", "BIDR", "RUB", "TUSD", "PAX", "DAI", "UAH", "ZAR", "IDRT", "NGN", "GYEN", "VAI")) {
                if (pair.endsWith(quote) && pair.length() > quote.length()) {
                    String base = pair.substring(0, pair.length() - quote.length());
                    result.computeIfAbsent(base, ignored -> new LinkedHashSet<>()).add(quote);
                    break;
                }
            }
        }
        return result;
    }

    private static KlineInterval parseInterval(String[] args, KlineInterval fallback) {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--interval".equals(args[i])) {
                return KlineInterval.parse(args[i + 1]);
            }
        }
        return fallback;
    }

    private static List<String> positionalPairs(String[] args) {
        List<String> values = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            if ("--interval".equals(args[i]) || "--cache-dir".equals(args[i]) || "--target-dir".equals(args[i])) {
                i++;
                continue;
            }
            values.add(args[i]);
        }
        if (values.size() == 2 && !values.get(0).contains("/")) {
            return List.of(values.get(0).toUpperCase() + "/" + values.get(1).toUpperCase());
        }
        List<String> pairs = new ArrayList<>();
        for (String value : values) {
            pairs.add(value.replace(':', '/').toUpperCase());
        }
        return pairs;
    }

    private static Path parsePathOption(String[] args, String option) {
        for (int i = 0; i < args.length - 1; i++) {
            if (option.equals(args[i])) {
                return Paths.get(args[i + 1]);
            }
        }
        return null;
    }

    private static void printHelp() {
        System.out.println("Usage: BinanceKlineCli <fetch|day|meta> ...");
        System.out.println("  fetch BTC USDT");
        System.out.println("  fetch BTC/USDT ETH/USDT --interval 1m");
        System.out.println("  day BTC USDT");
        System.out.println("  day BTC/USDT ETH/USDT --interval 1m");
        System.out.println("  meta BTC ETH SOL");
    }
}
