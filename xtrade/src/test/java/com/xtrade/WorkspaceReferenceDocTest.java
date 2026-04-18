package com.xtrade;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class WorkspaceReferenceDocTest {

    @Test
    void workspaceReferenceDocumentTracksUnifiedSystemAndArchiveStatus() throws IOException {
        Path doc = Path.of("docs", "workspace-reference.md");
        assertTrue(Files.exists(doc), "workspace reference doc should exist at docs/workspace-reference.md");

        String text = Files.readString(doc, StandardCharsets.UTF_8).toLowerCase();
        String[] requiredFragments = new String[] {
                "# unified workspace reference",
                "active modules",
                "xtrade",
                "binance",
                "archived components",
                "excluded from the active maven reactor",
                "drawthrucachingklinefeed",
                "klinefeedmonitorserver",
                "papertradingengine",
                "showdownharness",
                "cache hit rate",
                "feed latency",
                "producer connectivity",
                "prometheus"
        };

        for (String fragment : requiredFragments) {
            assertTrue(text.contains(fragment), "workspace reference doc should contain: " + fragment);
        }

        assertFalse(text.contains("mp/acapulco"), "active workspace doc should not reference retired mp paths");
        assertFalse(text.contains("mp/control"), "active workspace doc should not reference retired mp paths");
        assertFalse(text.contains("fetchklines.sh"), "active workspace doc should document xtrade tooling, not removed scripts");
    }
}
