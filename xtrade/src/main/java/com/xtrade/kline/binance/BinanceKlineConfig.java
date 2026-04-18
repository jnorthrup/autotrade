package com.xtrade.kline.binance;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Clock;
import java.util.Objects;
import java.util.Properties;

public final class BinanceKlineConfig {
    private static final String PROP_VISION_BASE_URL = "binance.vision.base-url";
    private static final String PROP_API_BASE_URL = "binance.api.base-url";
    private static final String PROP_START_YEAR = "binance.kline.start-year";
    private static final String PROP_RATE_LIMIT_MILLIS = "binance.kline.rate-limit-millis";
    private static final String PROP_CACHE_ROOT = "xtrade.kline.cache-root";
    private static final String PROP_IMPORT_ROOT = "xtrade.kline.import-root";
    private static final String ENV_CACHE_ROOT = "XTRADE_KLINE_CACHE_ROOT";
    private static final String ENV_IMPORT_ROOT = "XTRADE_KLINE_IMPORT_ROOT";
    private static final String ENV_XTRADE_BINANCE_VISION_BASE_URL = "XTRADE_BINANCE_VISION_BASE_URL";
    private static final String ENV_XTRADE_BINANCE_API_BASE_URL = "XTRADE_BINANCE_API_BASE_URL";
    private static final String ENV_XTRADE_BINANCE_START_YEAR = "XTRADE_BINANCE_START_YEAR";
    private static final String ENV_XTRADE_BINANCE_RATE_LIMIT_MILLIS = "XTRADE_BINANCE_RATE_LIMIT_MILLIS";

    private final URI visionBaseUrl;
    private final URI apiBaseUrl;
    private final int startYear;
    private final long rateLimitMillis;
    private final Path cacheRoot;
    private final Path importRoot;
    private final Clock clock;

    public BinanceKlineConfig(URI visionBaseUrl,
                              URI apiBaseUrl,
                              int startYear,
                              long rateLimitMillis,
                              Path cacheRoot,
                              Path importRoot,
                              Clock clock) {
        this.visionBaseUrl = Objects.requireNonNull(visionBaseUrl, "visionBaseUrl");
        this.apiBaseUrl = Objects.requireNonNull(apiBaseUrl, "apiBaseUrl");
        this.startYear = startYear;
        this.rateLimitMillis = rateLimitMillis;
        this.cacheRoot = Objects.requireNonNull(cacheRoot, "cacheRoot");
        this.importRoot = Objects.requireNonNull(importRoot, "importRoot");
        this.clock = Objects.requireNonNull(clock, "clock");
        if (startYear < 2017 || startYear > 3000) {
            throw new IllegalArgumentException("startYear out of range: " + startYear);
        }
        if (rateLimitMillis < 0) {
            throw new IllegalArgumentException("rateLimitMillis must be >= 0");
        }
    }

    public static BinanceKlineConfig load() {
        Properties props = loadProperties();
        String home = System.getProperty("user.home", ".");
        return new BinanceKlineConfig(
                uri(firstNonBlank(System.getenv(ENV_XTRADE_BINANCE_VISION_BASE_URL), props.getProperty(PROP_VISION_BASE_URL), "https://data.binance.vision")),
                uri(firstNonBlank(System.getenv(ENV_XTRADE_BINANCE_API_BASE_URL), props.getProperty(PROP_API_BASE_URL), "https://api.binance.com")),
                parseInt(firstNonBlank(System.getenv(ENV_XTRADE_BINANCE_START_YEAR), props.getProperty(PROP_START_YEAR), "2017"), 2017),
                parseLong(firstNonBlank(System.getenv(ENV_XTRADE_BINANCE_RATE_LIMIT_MILLIS), props.getProperty(PROP_RATE_LIMIT_MILLIS), "250"), 250L),
                expandPath(firstNonBlank(System.getenv(ENV_CACHE_ROOT), props.getProperty(PROP_CACHE_ROOT), home + "/xtrade-data/cache")),
                expandPath(firstNonBlank(System.getenv(ENV_IMPORT_ROOT), props.getProperty(PROP_IMPORT_ROOT), home + "/xtrade-data/import")),
                Clock.systemUTC());
    }

    public URI visionBaseUrl() {
        return visionBaseUrl;
    }

    public URI apiBaseUrl() {
        return apiBaseUrl;
    }

    public int startYear() {
        return startYear;
    }

    public long rateLimitMillis() {
        return rateLimitMillis;
    }

    public Path cacheRoot() {
        return cacheRoot;
    }

    public Path importRoot() {
        return importRoot;
    }

    public Clock clock() {
        return clock;
    }

    public Path defaultCacheDir(String tradeAsset, String counterAsset, String interval) {
        return cacheRoot.resolve(Paths.get("klines", interval, tradeAsset, counterAsset));
    }

    public Path defaultTargetDir(String tradeAsset, String counterAsset, String interval) {
        return importRoot.resolve(Paths.get("klines", interval, tradeAsset, counterAsset));
    }

    private static Properties loadProperties() {
        Properties properties = new Properties();
        try (InputStream in = BinanceKlineConfig.class.getClassLoader().getResourceAsStream("application.properties")) {
            if (in != null) {
                properties.load(in);
            }
        } catch (IOException ignored) {
        }
        return properties;
    }

    private static Path expandPath(String value) {
        String home = System.getProperty("user.home", ".");
        if (value.startsWith("~/")) {
            return Paths.get(home, value.substring(2));
        }
        return Paths.get(value);
    }

    private static URI uri(String value) {
        return URI.create(value.endsWith("/") ? value.substring(0, value.length() - 1) : value);
    }

    private static int parseInt(String value, int fallback) {
        try {
            return Integer.parseInt(value.trim());
        } catch (Exception e) {
            return fallback;
        }
    }

    private static long parseLong(String value, long fallback) {
        try {
            return Long.parseLong(value.trim());
        } catch (Exception e) {
            return fallback;
        }
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.trim().isEmpty()) {
                return value.trim();
            }
        }
        throw new IllegalArgumentException("no values supplied");
    }
}
