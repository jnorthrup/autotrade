package com.xtrade.kline;

import java.time.Duration;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Canonical interval registry shared by fetchers, muxers, agents, and paper trading consumers.
 */
public enum KlineInterval {
    ONE_SECOND("1s", Duration.ofSeconds(1)),
    FIVE_SECONDS("5s", Duration.ofSeconds(5)),
    FIFTEEN_SECONDS("15s", Duration.ofSeconds(15)),
    THIRTY_SECONDS("30s", Duration.ofSeconds(30)),
    ONE_MINUTE("1m", Duration.ofMinutes(1)),
    THREE_MINUTES("3m", Duration.ofMinutes(3)),
    FIVE_MINUTES("5m", Duration.ofMinutes(5)),
    FIFTEEN_MINUTES("15m", Duration.ofMinutes(15)),
    THIRTY_MINUTES("30m", Duration.ofMinutes(30)),
    ONE_HOUR("1h", Duration.ofHours(1)),
    TWO_HOURS("2h", Duration.ofHours(2)),
    FOUR_HOURS("4h", Duration.ofHours(4)),
    SIX_HOURS("6h", Duration.ofHours(6)),
    EIGHT_HOURS("8h", Duration.ofHours(8)),
    TWELVE_HOURS("12h", Duration.ofHours(12)),
    ONE_DAY("1d", Duration.ofDays(1)),
    THREE_DAYS("3d", Duration.ofDays(3)),
    ONE_WEEK("1w", Duration.ofDays(7)),
    ONE_MONTH("1M", Duration.ofDays(30));

    private static final Map<String, KlineInterval> BY_WIRE_NAME = buildLookup();

    private final String wireName;
    private final Duration duration;

    KlineInterval(String wireName, Duration duration) {
        this.wireName = wireName;
        this.duration = duration;
    }

    public String wireName() {
        return wireName;
    }

    public Duration duration() {
        return duration;
    }

    public long toMillis() {
        return duration.toMillis();
    }

    public long alignOpenTime(long epochMillis) {
        long bucket = toMillis();
        return Math.floorDiv(epochMillis, bucket) * bucket;
    }

    public long closeTimeExclusive(long openTimeMillis) {
        return openTimeMillis + toMillis();
    }

    public static KlineInterval parse(String wireName) {
        Objects.requireNonNull(wireName, "wireName must not be null");
        KlineInterval value = BY_WIRE_NAME.get(wireName);
        if (value == null) {
            throw new IllegalArgumentException("Unsupported kline interval: " + wireName);
        }
        return value;
    }

    private static Map<String, KlineInterval> buildLookup() {
        Map<String, KlineInterval> lookup = new LinkedHashMap<>();
        for (KlineInterval value : Arrays.asList(values())) {
            lookup.put(value.wireName, value);
        }
        return Collections.unmodifiableMap(lookup);
    }
}
