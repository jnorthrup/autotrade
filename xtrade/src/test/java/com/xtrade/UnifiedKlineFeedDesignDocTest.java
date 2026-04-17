package com.xtrade;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class UnifiedKlineFeedDesignDocTest {

    @Test
    void designDocumentExistsAndCoversSpiAndCachingSemantics() throws IOException {
        Path doc = Path.of("docs", "unified-kline-feed-design.md");
        assertTrue(Files.exists(doc), "design document should exist at docs/unified-kline-feed-design.md");

        String text = Files.readString(doc, StandardCharsets.UTF_8).toLowerCase();
        String[] requiredFragments = new String[] {
                "# unified kline model and draw-thru feed spi",
                "canonical kline bar schema",
                "venue",
                "symbol",
                "interval",
                "opentimemillis",
                "closetimemillis",
                "basevolume",
                "quotevolume",
                "takerbuybasevolume",
                "exchange metadata",
                "draw-thru caching feed",
                "backfill-on-demand",
                "streaming-forward",
                "producer spi",
                "registerproducer",
                "publish",
                "consumer spi",
                "requestbars",
                "subscribe",
                "papertradingengineklineadapter",
                "review checklist",
                "reviewed"
        };

        for (String fragment : requiredFragments) {
            assertTrue(text.contains(fragment), "design document should contain: " + fragment);
        }
    }
}
