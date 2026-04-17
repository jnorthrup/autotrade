package com.xtrade.kline;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Canonical immutable kline bar shared across producers and consumers.
 */
public final class KlineBar {
    private final KlineSeriesId seriesId;
    private final long openTimeMillis;
    private final long closeTimeMillis;
    private final long eventTimeMillis;
    private final long ingestTimeMillis;
    private final BigDecimal openPrice;
    private final BigDecimal highPrice;
    private final BigDecimal lowPrice;
    private final BigDecimal closePrice;
    private final BigDecimal baseVolume;
    private final BigDecimal quoteVolume;
    private final long tradeCount;
    private final BigDecimal takerBuyBaseVolume;
    private final BigDecimal takerBuyQuoteVolume;
    private final boolean closed;
    private final long sequence;
    private final String producerId;
    private final KlineSource source;
    private final Map<String, String> exchangeMetadata;

    public KlineBar(
            KlineSeriesId seriesId,
            long openTimeMillis,
            long closeTimeMillis,
            long eventTimeMillis,
            long ingestTimeMillis,
            BigDecimal openPrice,
            BigDecimal highPrice,
            BigDecimal lowPrice,
            BigDecimal closePrice,
            BigDecimal baseVolume,
            BigDecimal quoteVolume,
            long tradeCount,
            BigDecimal takerBuyBaseVolume,
            BigDecimal takerBuyQuoteVolume,
            boolean closed,
            long sequence,
            String producerId,
            KlineSource source,
            Map<String, String> exchangeMetadata) {
        this.seriesId = Objects.requireNonNull(seriesId, "seriesId must not be null");
        this.openTimeMillis = openTimeMillis;
        this.closeTimeMillis = closeTimeMillis;
        this.eventTimeMillis = eventTimeMillis;
        this.ingestTimeMillis = ingestTimeMillis;
        this.openPrice = positive(openPrice, "openPrice");
        this.highPrice = positive(highPrice, "highPrice");
        this.lowPrice = positive(lowPrice, "lowPrice");
        this.closePrice = positive(closePrice, "closePrice");
        this.baseVolume = nonNegative(baseVolume, "baseVolume");
        this.quoteVolume = nonNegative(quoteVolume, "quoteVolume");
        this.tradeCount = tradeCount;
        this.takerBuyBaseVolume = nonNegative(takerBuyBaseVolume, "takerBuyBaseVolume");
        this.takerBuyQuoteVolume = nonNegative(takerBuyQuoteVolume, "takerBuyQuoteVolume");
        this.closed = closed;
        this.sequence = sequence;
        this.producerId = requireText(producerId, "producerId");
        this.source = Objects.requireNonNull(source, "source must not be null");
        this.exchangeMetadata = immutableCopy(exchangeMetadata);
        validate();
    }

    public KlineSeriesId seriesId() {
        return seriesId;
    }

    public long openTimeMillis() {
        return openTimeMillis;
    }

    public long closeTimeMillis() {
        return closeTimeMillis;
    }

    public long eventTimeMillis() {
        return eventTimeMillis;
    }

    public long ingestTimeMillis() {
        return ingestTimeMillis;
    }

    public BigDecimal openPrice() {
        return openPrice;
    }

    public BigDecimal highPrice() {
        return highPrice;
    }

    public BigDecimal lowPrice() {
        return lowPrice;
    }

    public BigDecimal closePrice() {
        return closePrice;
    }

    public BigDecimal baseVolume() {
        return baseVolume;
    }

    public BigDecimal quoteVolume() {
        return quoteVolume;
    }

    public long tradeCount() {
        return tradeCount;
    }

    public BigDecimal takerBuyBaseVolume() {
        return takerBuyBaseVolume;
    }

    public BigDecimal takerBuyQuoteVolume() {
        return takerBuyQuoteVolume;
    }

    public boolean closed() {
        return closed;
    }

    public long sequence() {
        return sequence;
    }

    public String producerId() {
        return producerId;
    }

    public KlineSource source() {
        return source;
    }

    public Map<String, String> exchangeMetadata() {
        return exchangeMetadata;
    }

    public String metadata(String key) {
        return exchangeMetadata.get(key);
    }

    public boolean isFinal() {
        return closed;
    }

    private void validate() {
        if (openTimeMillis < 0 || closeTimeMillis < 0 || eventTimeMillis < 0 || ingestTimeMillis < 0) {
            throw new IllegalArgumentException("timestamps must be non-negative");
        }
        long intervalMillis = seriesId.interval().toMillis();
        if (closeTimeMillis <= openTimeMillis) {
            throw new IllegalArgumentException("closeTimeMillis must be greater than openTimeMillis");
        }
        if (closeTimeMillis - openTimeMillis != intervalMillis) {
            throw new IllegalArgumentException("bar width must equal interval width for " + seriesId.interval().wireName());
        }
        if (seriesId.interval().alignOpenTime(openTimeMillis) != openTimeMillis) {
            throw new IllegalArgumentException("openTimeMillis is not aligned to interval boundary");
        }
        if (highPrice.compareTo(openPrice) < 0 || highPrice.compareTo(closePrice) < 0 || highPrice.compareTo(lowPrice) < 0) {
            throw new IllegalArgumentException("highPrice must be >= open, close, and low");
        }
        if (lowPrice.compareTo(openPrice) > 0 || lowPrice.compareTo(closePrice) > 0) {
            throw new IllegalArgumentException("lowPrice must be <= open and close");
        }
        if (tradeCount < 0) {
            throw new IllegalArgumentException("tradeCount must not be negative");
        }
        if (eventTimeMillis < openTimeMillis) {
            throw new IllegalArgumentException("eventTimeMillis must be >= openTimeMillis");
        }
        if (ingestTimeMillis < openTimeMillis) {
            throw new IllegalArgumentException("ingestTimeMillis must be >= openTimeMillis");
        }
    }

    private static Map<String, String> immutableCopy(Map<String, String> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return Collections.emptyMap();
        }
        Map<String, String> copy = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : metadata.entrySet()) {
            copy.put(requireText(entry.getKey(), "exchangeMetadata key"),
                    requireText(entry.getValue(), "exchangeMetadata value"));
        }
        return Collections.unmodifiableMap(copy);
    }

    private static BigDecimal positive(BigDecimal value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        if (value.signum() <= 0) {
            throw new IllegalArgumentException(field + " must be positive");
        }
        return value;
    }

    private static BigDecimal nonNegative(BigDecimal value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        if (value.signum() < 0) {
            throw new IllegalArgumentException(field + " must not be negative");
        }
        return value;
    }

    private static String requireText(String value, String field) {
        Objects.requireNonNull(value, field + " must not be null");
        String trimmed = value.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return trimmed;
    }
}
