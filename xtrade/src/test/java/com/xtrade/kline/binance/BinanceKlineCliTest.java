package com.xtrade.kline.binance;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BinanceKlineCliTest {

    @Test
    void metaCatalogLivesInsideXtradeResources() throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream original = System.out;
        try {
            System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8.name()));
            assertEquals(0, BinanceKlineCli.run(new String[]{"meta", "BTC", "ETH"}));
        } finally {
            System.setOut(original);
        }

        String text = output.toString(StandardCharsets.UTF_8);
        assertTrue(text.contains("binance-kline fetch BTC "));
        assertTrue(text.contains("binance-kline fetch ETH "));
        assertTrue(text.contains("USDT"));
    }

    @Test
    void suffixParserBuildsQuoteSetsFromCatalogTokens() {
        Map<String, Set<String>> suffixes = BinanceKlineCli.parseSuffixes("BTCUSDT\nBTCBUSD\nETHBTC\nINVALID-ROW\n");

        assertEquals(Set.of("USDT", "BUSD"), suffixes.get("BTC"));
        assertEquals(Set.of("BTC"), suffixes.get("ETH"));
    }
}
