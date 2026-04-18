package com.xtrade;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ReadmeOperationalGuideTest {

    @Test
    void readmeDocumentsUnifiedFeedConfigurationAndOperations() throws IOException {
        Path readme = Path.of("README.md");
        assertTrue(Files.exists(readme), "README should exist");

        String text = Files.readString(readme, StandardCharsets.UTF_8).toLowerCase();
        String[] requiredFragments = new String[] {
                "unified kline architecture",
                "configuration",
                "xtrade.kline.cache-root",
                "xtrade.kline.import-root",
                "feed health and metrics",
                "/health",
                "/metrics",
                "cache hit rate",
                "feed latency",
                "producer connectivity",
                "stale-producer alerts",
                "klinefeedmonitorserver",
                "kline feed operations runbook",
                "archived legacy workspace"
        };

        for (String fragment : requiredFragments) {
            assertTrue(text.contains(fragment), "README should contain: " + fragment);
        }
    }
}
