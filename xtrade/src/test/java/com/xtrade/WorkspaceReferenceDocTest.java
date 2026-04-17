package com.xtrade;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class WorkspaceReferenceDocTest {

    @Test
    void workspaceReferenceDocumentExistsAndCoversRequiredSections() throws IOException {
        Path doc = Path.of("docs", "workspace-reference.md");
        assertTrue(Files.exists(doc), "workspace reference doc should exist at docs/workspace-reference.md");

        String text = Files.readString(doc, StandardCharsets.UTF_8);
        String lower = text.toLowerCase();

        String[] requiredFragments = new String[] {
                "## 2) complete workspace inventory",
                "mp/acapulco",
                "mp/control",
                "mp/databinance",
                "mp/money",
                "mp/trikeshed-adapter",
                "mp/bin",
                "fetchklines.sh",
                "dayklines.sh",
                "fetchtrades.sh",
                "meta-klines.sh",
                "allcachedpairs.sh",
                "tweeze.sh",
                "vizgnome.sh",
                "shardnode.sh",
                "keypair.sh",
                "apikeyserver.sh",
                "cmcsandboxdata.sh",
                "simawallet concepts",
                "paper trade lifecycle",
                "position tracking",
                "xchange framework integration surface",
                "exchangeservice.getalltickers",
                "marketdataserviceimpl.fetchalltickers",
                "papertradingengine",
                "databinancevision.klines"
        };

        for (String fragment : requiredFragments) {
            assertTrue(lower.contains(fragment), "workspace reference doc should contain: " + fragment);
        }
    }
}
