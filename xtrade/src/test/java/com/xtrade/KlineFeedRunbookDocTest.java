package com.xtrade;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class KlineFeedRunbookDocTest {

    @Test
    void runbookDocumentsHealthMetricsAndStalenessAlerts() throws IOException {
        Path doc = Path.of("docs", "kline-feed-operations-runbook.md");
        assertTrue(Files.exists(doc), "runbook should exist");

        String text = Files.readString(doc, StandardCharsets.UTF_8).toLowerCase();
        String[] requiredFragments = new String[] {
                "health checks",
                "/health",
                "/metrics",
                "cache hit rate",
                "feed latency",
                "producer connectivity",
                "feed staleness alert",
                "triage flow for stale feed alerts",
                "recovery verification"
        };

        for (String fragment : requiredFragments) {
            assertTrue(text.contains(fragment), "runbook should contain: " + fragment);
        }
    }
}
