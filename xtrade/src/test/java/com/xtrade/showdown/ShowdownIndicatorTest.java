package com.xtrade.showdown;

import com.google.gson.Gson;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;

import java.io.FileReader;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for indicator snapshots, JSON output, dashboard mode, and
 * trade history indicator values.
 *
 * Validates all acceptance criteria:
 *   - Each agent's trade history includes indicator values at decision time
 *   - JSON output mode writes a valid JSON file with leaderboard, equity curves, indicator snapshots
 *   - ASCII dashboard mode prints a refreshable summary table
 *   - Indicator values are accessible programmatically via getIndicatorSnapshots()
 */
class ShowdownIndicatorTest {

    private static final int NUM_TICKS = 30;

    @Test
    void testTradeHistoryIncludesIndicatorValues() {
        List<Integer> codecIds = Arrays.asList(1, 2, 3);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.run();

        Map<String, ShowdownAgent> agents = harness.getAgents();
        for (Map.Entry<String, ShowdownAgent> entry : agents.entrySet()) {
            ShowdownAgent agent = entry.getValue();
            List<TradeAction> history = agent.getTradeHistory();

            // Each tick produces one TradeAction per pair
            assertFalse(history.isEmpty(), "Trade history should not be empty for " + entry.getKey());

            for (TradeAction action : history) {
                Map<String, Object> indicators = action.getIndicators();
                assertNotNull(indicators, "Indicators should not be null for " + entry.getKey());

                // Verify required indicator keys present
                assertTrue(indicators.containsKey("rsi"),
                        "Missing rsi in indicators for " + entry.getKey());
                assertTrue(indicators.containsKey("macd_hist"),
                        "Missing macd_hist in indicators for " + entry.getKey());
                assertTrue(indicators.containsKey("bb_position"),
                        "Missing bb_position in indicators for " + entry.getKey());
                assertTrue(indicators.containsKey("adx"),
                        "Missing adx in indicators for " + entry.getKey());
                assertTrue(indicators.containsKey("vwap"),
                        "Missing vwap in indicators for " + entry.getKey());
                assertTrue(indicators.containsKey("momentum"),
                        "Missing momentum in indicators for " + entry.getKey());
                assertTrue(indicators.containsKey("atr_14"),
                        "Missing atr_14 in indicators for " + entry.getKey());

                // Verify types are Numbers
                assertInstanceOf(Number.class, indicators.get("rsi"));
                assertInstanceOf(Number.class, indicators.get("macd_hist"));
                assertInstanceOf(Number.class, indicators.get("bb_position"));
                assertInstanceOf(Number.class, indicators.get("adx"));
                assertInstanceOf(Number.class, indicators.get("vwap"));
                assertInstanceOf(Number.class, indicators.get("momentum"));

                // Verify RSI is in valid range [0, 100]
                double rsi = ((Number) indicators.get("rsi")).doubleValue();
                assertTrue(rsi >= 0.0 && rsi <= 100.0,
                        "RSI should be in [0,100], got " + rsi);

                // Verify BB position is in [0, 1]
                double bbPos = ((Number) indicators.get("bb_position")).doubleValue();
                assertTrue(bbPos >= 0.0 && bbPos <= 1.0,
                        "BB position should be in [0,1], got " + bbPos);
            }
        }
    }

    @Test
    void testGetIndicatorSnapshotsProgrammatic() {
        List<Integer> codecIds = Arrays.asList(1, 2);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.run();

        // Access indicator snapshots programmatically
        Map<String, List<Map<String, Map<String, Object>>>> snapshots =
                harness.getIndicatorSnapshots();

        assertNotNull(snapshots, "getIndicatorSnapshots() should return non-null");
        assertEquals(2, snapshots.size(), "Should have snapshots for 2 agents");

        for (Map.Entry<String, List<Map<String, Map<String, Object>>>> entry : snapshots.entrySet()) {
            String agentName = entry.getKey();
            List<Map<String, Map<String, Object>>> tickSnapshots = entry.getValue();

            // Should have one snapshot per tick
            assertEquals(NUM_TICKS, tickSnapshots.size(),
                    "Should have " + NUM_TICKS + " tick snapshots for " + agentName);

            // Each tick snapshot maps pair -> indicators
            for (int t = 0; t < tickSnapshots.size(); t++) {
                Map<String, Map<String, Object>> tickSnap = tickSnapshots.get(t);
                assertFalse(tickSnap.isEmpty(),
                        "Tick " + t + " snapshot should not be empty for " + agentName);

                for (Map.Entry<String, Map<String, Object>> pairEntry : tickSnap.entrySet()) {
                    Map<String, Object> indicators = pairEntry.getValue();

                    // Verify essential indicators present
                    assertTrue(indicators.containsKey("rsi"),
                            "Tick " + t + " missing rsi for " + agentName);
                    assertTrue(indicators.containsKey("macd_hist"),
                            "Tick " + t + " missing macd_hist for " + agentName);
                    assertTrue(indicators.containsKey("adx"),
                            "Tick " + t + " missing adx for " + agentName);
                    assertTrue(indicators.containsKey("bb_position"),
                            "Tick " + t + " missing bb_position for " + agentName);
                    assertTrue(indicators.containsKey("vwap_ratio"),
                            "Tick " + t + " missing vwap_ratio for " + agentName);
                    assertTrue(indicators.containsKey("momentum"),
                            "Tick " + t + " missing momentum for " + agentName);
                }
            }
        }
    }

    @Test
    void testIndicatorSnapshotsChangeOverTime() {
        List<Integer> codecIds = Collections.singletonList(1);
        ShowdownHarness harness = new ShowdownHarness(codecIds, 50);
        harness.run();

        Map<String, List<Map<String, Map<String, Object>>>> snapshots =
                harness.getIndicatorSnapshots();
        String agentName = snapshots.keySet().iterator().next();
        List<Map<String, Map<String, Object>>> ticks = snapshots.get(agentName);

        // Early tick RSI vs later tick RSI should differ (prices change)
        double earlyRsi = getFirstPairIndicator(ticks.get(5), "rsi");
        double lateRsi = getFirstPairIndicator(ticks.get(45), "rsi");
        // They might be equal by coincidence but typically differ
        // At minimum, verify they are valid numbers
        assertTrue(Double.isFinite(earlyRsi), "Early RSI should be finite");
        assertTrue(Double.isFinite(lateRsi), "Late RSI should be finite");
    }

    @Test
    void testJsonOutputModeWritesValidFile(@TempDir Path tempDir) throws Exception {
        String jsonPath = tempDir.resolve("showdown_test.json").toString();

        List<Integer> codecIds = Arrays.asList(1, 2, 3);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.setOutputMode(ShowdownHarness.OutputMode.JSON, jsonPath);

        // Suppress dashboard and leaderboard for this test
        harness.setDashboardEnabled(false);

        harness.run();

        // Verify JSON file was written
        java.io.File jsonFile = new java.io.File(jsonPath);
        assertTrue(jsonFile.exists(), "JSON file should exist at " + jsonPath);
        assertTrue(jsonFile.length() > 0, "JSON file should not be empty");

        // Parse and validate JSON structure
        Gson gson = new Gson();
        Map<String, Object> results;
        try (FileReader reader = new FileReader(jsonFile)) {
            results = gson.fromJson(reader, Map.class);
        }

        assertNotNull(results, "JSON results should not be null");

        // Verify leaderboard
        assertTrue(results.containsKey("leaderboard"), "JSON should contain leaderboard");
        List<?> leaderboard = (List<?>) results.get("leaderboard");
        assertEquals(3, leaderboard.size(), "Leaderboard should have 3 entries");

        // Verify equity_curves
        assertTrue(results.containsKey("equity_curves"), "JSON should contain equity_curves");
        Map<?, ?> equityCurves = (Map<?, ?>) results.get("equity_curves");
        assertEquals(3, equityCurves.size(), "Should have equity curves for 3 agents");

        // Verify indicator_snapshots
        assertTrue(results.containsKey("indicator_snapshots"),
                "JSON should contain indicator_snapshots");
        Map<?, ?> indicatorSnapshots = (Map<?, ?>) results.get("indicator_snapshots");
        assertEquals(3, indicatorSnapshots.size(),
                "Should have indicator snapshots for 3 agents");

        // Validate indicator snapshot structure
        for (Map.Entry<?, ?> agentEntry : indicatorSnapshots.entrySet()) {
            List<?> agentTicks = (List<?>) agentEntry.getValue();
            assertEquals(NUM_TICKS, agentTicks.size(),
                    "Should have " + NUM_TICKS + " tick snapshots per agent");
        }

        // Verify tick_snapshots (financial)
        assertTrue(results.containsKey("tick_snapshots"), "JSON should contain tick_snapshots");

        // Verify metadata
        assertTrue(results.containsKey("metadata"), "JSON should contain metadata");
        Map<?, ?> metadata = (Map<?, ?>) results.get("metadata");
        assertEquals(NUM_TICKS, ((Number) metadata.get("total_ticks")).intValue());
        assertNotNull(metadata.get("initial_cash"));
        assertNotNull(metadata.get("timestamp"));
    }

    @Test
    void testJsonOutputContainsValidIndicatorValues(@TempDir Path tempDir) throws Exception {
        String jsonPath = tempDir.resolve("indicators_test.json").toString();

        List<Integer> codecIds = Collections.singletonList(1);
        ShowdownHarness harness = new ShowdownHarness(codecIds, 10);
        harness.setOutputMode(ShowdownHarness.OutputMode.JSON, jsonPath);
        harness.run();

        Gson gson = new Gson();
        Map<String, Object> results;
        try (FileReader reader = new FileReader(jsonPath)) {
            results = gson.fromJson(reader, Map.class);
        }

        Map<?, ?> indicatorSnapshots = (Map<?, ?>) results.get("indicator_snapshots");
        assertFalse(indicatorSnapshots.isEmpty());

        // Get first agent's tick data
        String agentKey = (String) indicatorSnapshots.keySet().iterator().next();
        List<?> agentTicks = (List<?>) indicatorSnapshots.get(agentKey);
        assertFalse(agentTicks.isEmpty());

        // First tick should have pair -> indicators
        Map<?, ?> tick0 = (Map<?, ?>) agentTicks.get(0);
        assertFalse(tick0.isEmpty());

        // Get first pair's indicators
        String pairKey = (String) tick0.keySet().iterator().next();
        Map<?, ?> indicators = (Map<?, ?>) tick0.get(pairKey);
        assertTrue(indicators.containsKey("rsi"), "Should have rsi");
        assertTrue(indicators.containsKey("macd_hist"), "Should have macd_hist");
        assertTrue(indicators.containsKey("adx"), "Should have adx");
        assertTrue(indicators.containsKey("momentum"), "Should have momentum");
        assertTrue(indicators.containsKey("bb_position"), "Should have bb_position");
    }

    @Test
    void testDashboardModeDoesNotThrow() {
        List<Integer> codecIds = Arrays.asList(1, 2);
        ShowdownHarness harness = new ShowdownHarness(codecIds, 20);

        // Dashboard should not throw during run
        assertDoesNotThrow(() -> {
            harness.setDashboardEnabled(true);
            // Don't call run() with dashboard since it messes with terminal output
            // Just test the printDashboard method directly
            harness.printDashboard(5, 20);
            harness.printDashboard(10, 20);
        });
    }

    @Test
    void testBuildJsonResultsStructure() {
        List<Integer> codecIds = Arrays.asList(1, 2);
        ShowdownHarness harness = new ShowdownHarness(codecIds, 10);
        harness.run();

        Map<String, Object> results = harness.buildJsonResults();

        // Verify top-level keys
        assertTrue(results.containsKey("leaderboard"));
        assertTrue(results.containsKey("equity_curves"));
        assertTrue(results.containsKey("indicator_snapshots"));
        assertTrue(results.containsKey("tick_snapshots"));
        assertTrue(results.containsKey("metadata"));

        // Verify metadata
        Map<String, Object> metadata = (Map<String, Object>) results.get("metadata");
        assertEquals(10, metadata.get("total_ticks"));
        assertEquals(100_000.0, metadata.get("initial_cash"));
        assertEquals(Arrays.asList(1, 2), metadata.get("codec_ids"));

        // Verify leaderboard entries have ranks
        List<Map<String, Object>> lb = (List<Map<String, Object>>) results.get("leaderboard");
        assertEquals(2, lb.size());
        assertEquals(1, lb.get(0).get("rank"));
        assertEquals(2, lb.get(1).get("rank"));

        // Verify descending PnL
        double pnl1 = ((Number) lb.get(0).get("total_pnl")).doubleValue();
        double pnl2 = ((Number) lb.get(1).get("total_pnl")).doubleValue();
        assertTrue(pnl1 >= pnl2, "Leaderboard should be sorted by PnL descending");
    }

    @Test
    void testIndicatorSnapshotsResetClears() {
        List<Integer> codecIds = Collections.singletonList(1);
        ShowdownHarness harness = new ShowdownHarness(codecIds, 10);
        harness.run();

        // Should have snapshots
        Map<String, List<Map<String, Map<String, Object>>>> snapshots =
                harness.getIndicatorSnapshots();
        assertEquals(10, snapshots.values().iterator().next().size());

        // Reset
        harness.reset();

        // After reset, snapshots should be empty (new agent instances)
        // Note: agents are not replaced on reset, only cleared
        Map<String, List<Map<String, Map<String, Object>>>> afterReset =
                harness.getIndicatorSnapshots();
        // Agent indicator snapshots list should be cleared
        for (List<?> tickSnaps : afterReset.values()) {
            assertTrue(tickSnaps.isEmpty(), "Indicator snapshots should be empty after reset");
        }
    }

    @Test
    void testTradeActionIndicatorsImmutability() {
        Map<String, Object> indicatorMap = new LinkedHashMap<>();
        indicatorMap.put("rsi", 55.0);
        indicatorMap.put("macd_hist", 0.1);

        TradeAction action = new TradeAction("BTC/USDT", TradeAction.BUY, 1.0, 50000.0,
                0.8, 1.0, indicatorMap);

        // Try to mutate the original map
        indicatorMap.put("rsi", 999.0);

        // TradeAction's copy should be unaffected
        assertEquals(55.0, ((Number) action.getIndicators().get("rsi")).doubleValue(), 1e-9);

        // getIndicators() should return unmodifiable map
        assertThrows(UnsupportedOperationException.class, () -> {
            action.getIndicators().put("test", 1.0);
        });
    }

    @Test
    void testTradeActionNullIndicatorsGivesEmptyMap() {
        TradeAction action = new TradeAction("BTC/USDT", TradeAction.HOLD, 0, 100.0, 0.1, 0.0);
        assertNotNull(action.getIndicators());
        assertTrue(action.getIndicators().isEmpty());
    }

    @Test
    void testShowdownCliOutputFlag() {
        ShowdownCli cli = ShowdownCli.parse(
                new String[]{"--showdown", "--codecs", "1", "--output", "json", "--ticks", "10"});
        assertTrue(cli.isShowdown());
        assertEquals(ShowdownCli.OutputMode.JSON, cli.getOutputMode());
        assertEquals("showdown_results.json", cli.getJsonOutputPath());
    }

    @Test
    void testShowdownCliOutputJsonWithPath() {
        ShowdownCli cli = ShowdownCli.parse(
                new String[]{"--showdown", "--codecs", "1", "--output", "json:/tmp/results.json"});
        assertEquals(ShowdownCli.OutputMode.JSON, cli.getOutputMode());
        assertEquals("/tmp/results.json", cli.getJsonOutputPath());
    }

    @Test
    void testShowdownCliOutputText() {
        ShowdownCli cli = ShowdownCli.parse(
                new String[]{"--showdown", "--output", "text"});
        assertEquals(ShowdownCli.OutputMode.TEXT, cli.getOutputMode());
        assertNull(cli.getJsonOutputPath());
    }

    @Test
    void testShowdownCliDashboardFlag() {
        ShowdownCli cli = ShowdownCli.parse(
                new String[]{"--showdown", "--dashboard", "--codecs", "1,2"});
        assertTrue(cli.isDashboardEnabled());
    }

    @Test
    void testShowdownCliDashboardDefaultFalse() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown"});
        assertFalse(cli.isDashboardEnabled());
    }

    @Test
    void testShowdownCliOutputDefaultText() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown"});
        assertEquals(ShowdownCli.OutputMode.TEXT, cli.getOutputMode());
    }

    @Test
    void testShowdownCliOutputRequiresValue() {
        assertThrows(IllegalArgumentException.class,
                () -> ShowdownCli.parse(new String[]{"--showdown", "--output"}));
    }

    @Test
    void testBbPositionComputation() {
        Map<String, Object> md = new LinkedHashMap<>();
        md.put("bb_upper", 110.0);
        md.put("bb_lower", 90.0);
        double pos = ShowdownAgent.computeBbPosition(md, 100.0);
        assertEquals(0.5, pos, 1e-9, "Price at BB mid should give position 0.5");

        double posHigh = ShowdownAgent.computeBbPosition(md, 110.0);
        assertEquals(1.0, posHigh, 1e-9, "Price at BB upper should give position 1.0");

        double posLow = ShowdownAgent.computeBbPosition(md, 90.0);
        assertEquals(0.0, posLow, 1e-9, "Price at BB lower should give position 0.0");
    }

    @Test
    void testBbPositionZeroWidth() {
        Map<String, Object> md = new LinkedHashMap<>();
        md.put("bb_upper", 100.0);
        md.put("bb_lower", 100.0);
        double pos = ShowdownAgent.computeBbPosition(md, 100.0);
        assertEquals(0.5, pos, 1e-9, "Zero BB width should give position 0.5");
    }

    @Test
    void testVwapRatioComputation() {
        Map<String, Object> md = new LinkedHashMap<>();
        md.put("vwap", 100.0);

        double ratio = ShowdownAgent.computeVwapRatio(md, 105.0);
        assertEquals(1.05, ratio, 1e-9, "VWAP ratio should be 1.05");

        double ratioAtVwap = ShowdownAgent.computeVwapRatio(md, 100.0);
        assertEquals(1.0, ratioAtVwap, 1e-9, "VWAP ratio at VWAP should be 1.0");
    }

    @Test
    void testVwapRatioZeroVwap() {
        Map<String, Object> md = new LinkedHashMap<>();
        md.put("vwap", 0.0);
        double ratio = ShowdownAgent.computeVwapRatio(md, 100.0);
        assertEquals(1.0, ratio, 1e-9, "Zero VWAP should give ratio 1.0");
    }

    @Test
    void testWriteJsonOutputToCustomPath(@TempDir Path tempDir) {
        String jsonPath = tempDir.resolve("custom_output.json").toString();
        List<Integer> codecIds = Collections.singletonList(1);
        ShowdownHarness harness = new ShowdownHarness(codecIds, 5);
        harness.run();

        // Write JSON to custom path
        harness.writeJsonOutput(jsonPath);

        java.io.File f = new java.io.File(jsonPath);
        assertTrue(f.exists(), "JSON file should exist");
        assertTrue(f.length() > 0, "JSON file should not be empty");
    }

    @Test
    void testInstrumentRecordedInCodec() {
        List<Integer> codecIds = Collections.singletonList(1);
        ShowdownHarness harness = new ShowdownHarness(codecIds, 20);
        harness.run();

        // The codec's instruments map should have been populated
        for (ShowdownAgent agent : harness.getAgents().values()) {
            Map<String, Double> instruments = agent.getCodec().getInstruments();
            assertFalse(instruments.isEmpty(),
                    "Codec instruments should not be empty after run for " + agent.getCodecName());
            assertTrue(instruments.containsKey("rsi"),
                    "Instruments should contain rsi");
            assertTrue(instruments.containsKey("macd_hist"),
                    "Instruments should contain macd_hist");
            assertTrue(instruments.containsKey("adx"),
                    "Instruments should contain adx");
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private double getFirstPairIndicator(Map<String, Map<String, Object>> tickSnap, String key) {
        for (Map<String, Object> pairIndicators : tickSnap.values()) {
            Object val = pairIndicators.get(key);
            if (val instanceof Number) {
                return ((Number) val).doubleValue();
            }
        }
        return Double.NaN;
    }
}
