package com.xtrade.kline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Health report built from a metrics snapshot plus staleness alerts.
 */
public final class KlineFeedHealthReport {
    private final String status;
    private final long generatedAtMillis;
    private final KlineFeedMetricsSnapshot metrics;
    private final List<String> alerts;

    public KlineFeedHealthReport(String status,
                                 long generatedAtMillis,
                                 KlineFeedMetricsSnapshot metrics,
                                 List<String> alerts) {
        this.status = requireText(status, "status");
        this.generatedAtMillis = generatedAtMillis;
        this.metrics = Objects.requireNonNull(metrics, "metrics must not be null");
        this.alerts = Collections.unmodifiableList(new ArrayList<>(alerts));
    }

    public String status() {
        return status;
    }

    public long generatedAtMillis() {
        return generatedAtMillis;
    }

    public KlineFeedMetricsSnapshot metrics() {
        return metrics;
    }

    public List<String> alerts() {
        return alerts;
    }

    public boolean healthy() {
        return alerts.isEmpty() && "OK".equals(status);
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
