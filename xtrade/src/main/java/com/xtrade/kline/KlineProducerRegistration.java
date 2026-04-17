package com.xtrade.kline;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Producer registration advertised to the feed.
 */
public final class KlineProducerRegistration {
    private final String producerId;
    private final String producerName;
    private final Set<KlineSeriesId> publishedSeries;
    private final KlineBackfillProvider backfillProvider;

    public KlineProducerRegistration(String producerId,
                                     String producerName,
                                     Collection<KlineSeriesId> publishedSeries,
                                     KlineBackfillProvider backfillProvider) {
        this.producerId = requireText(producerId, "producerId");
        this.producerName = requireText(producerName, "producerName");
        Objects.requireNonNull(publishedSeries, "publishedSeries must not be null");
        if (publishedSeries.isEmpty()) {
            throw new IllegalArgumentException("publishedSeries must not be empty");
        }
        LinkedHashSet<KlineSeriesId> copy = new LinkedHashSet<>(publishedSeries);
        this.publishedSeries = Collections.unmodifiableSet(copy);
        this.backfillProvider = backfillProvider;
    }

    public String producerId() {
        return producerId;
    }

    public String producerName() {
        return producerName;
    }

    public Set<KlineSeriesId> publishedSeries() {
        return publishedSeries;
    }

    public KlineBackfillProvider backfillProvider() {
        return backfillProvider;
    }

    public boolean supports(KlineSeriesId seriesId) {
        return publishedSeries.contains(seriesId);
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
