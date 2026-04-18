package com.xtrade.showdown;

import com.xtrade.codec.CodecRegistry;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Timeout;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end acceptance test for the full 24-agent showdown.
 *
 * Validates all acceptance criteria:
 *   - Showdown runs 24 agents for 500 ticks and completes in under 60 seconds
 *   - Leaderboard output contains exactly 24 ranked agents with valid metrics
 *   - All agents produce at least 1 trade over 500 ticks (no degenerate zero-trade codecs)
 *   - All summary metrics are valid (non-NaN, correct types)
 *
 * This test uses the programmatic API (ShowdownHarness) to avoid needing
 * a pre-built JAR, making it compatible with {@code mvn verify}.
 */
@DisplayName("Showdown Acceptance Test — Full 24-agent, 500-tick E2E")
class ShowdownAcceptanceTest {

    private static final int NUM_CODECS = 24;
    private static final int NUM_TICKS = 500;

    @Test
    @DisplayName("Full showdown: 24 agents, 500 ticks, all codecs trade")
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void testFullShowdownAllCodecs() {
        // 1. Build codec IDs 1-24
        List<Integer> codecIds = new ArrayList<>();
        for (int i = 1; i <= NUM_CODECS; i++) {
            codecIds.add(i);
        }
        assertEquals(NUM_CODECS, codecIds.size(), "Should prepare 24 codec IDs");

        // 2. Create and run showdown harness
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        long startTime = System.currentTimeMillis();
        Map<String, Map<String, Object>> results = harness.run();
        long elapsed = System.currentTimeMillis() - startTime;

        System.out.printf("[Acceptance] Showdown completed in %d ms%n", elapsed);

        // 3. Validate it completed in under 60 seconds
        assertTrue(elapsed < 60_000,
                "Showdown should complete in under 60 seconds, took: " + elapsed + " ms");

        // 4. Validate exactly 24 agents in results
        assertNotNull(results, "Results map should not be null");
        assertEquals(NUM_CODECS, results.size(),
                "Leaderboard should contain exactly 24 agents, found: " + results.size());

        // 5. Validate leaderboard ordering
        List<Map<String, Object>> leaderboard = harness.getLeaderboard();
        assertEquals(NUM_CODECS, leaderboard.size(),
                "Leaderboard should have exactly 24 entries");

        // 6. Validate each agent in the leaderboard
        Set<Integer> seenRanks = new HashSet<>();
        for (int i = 0; i < leaderboard.size(); i++) {
            Map<String, Object> entry = leaderboard.get(i);

            // Rank should be sequential from 1 to 24
            int rank = ((Number) entry.get("rank")).intValue();
            assertTrue(rank >= 1 && rank <= NUM_CODECS,
                    "Rank should be in [1,24], got: " + rank);
            assertFalse(seenRanks.contains(rank), "Duplicate rank: " + rank);
            seenRanks.add(rank);

            // Agent name should be present
            String agentName = (String) entry.get("agent_name");
            assertNotNull(agentName, "Agent name should not be null");
            assertFalse(agentName.isEmpty(), "Agent name should not be empty");

            // All required metrics must be present and valid
            assertValidMetrics(entry, agentName);

            // Trade count must be >= 1 (no zero-trade codecs)
            int tradeCount = ((Number) entry.get("trade_count")).intValue();
            assertTrue(tradeCount >= 1,
                    "Agent " + agentName + " should have at least 1 trade, got: " + tradeCount);
        }

        // 7. Validate ranks are 1..24
        assertEquals(NUM_CODECS, seenRanks.size(), "Should have 24 unique ranks");

        // 8. Validate descending PnL order in leaderboard
        for (int i = 0; i < leaderboard.size() - 1; i++) {
            double pnlA = ((Number) leaderboard.get(i).get("total_pnl")).doubleValue();
            double pnlB = ((Number) leaderboard.get(i + 1).get("total_pnl")).doubleValue();
            assertTrue(pnlA >= pnlB,
                    "Leaderboard should be descending by PnL: " + pnlA + " >= " + pnlB
                            + " at positions " + i + "," + (i + 1));
        }

        // 9. Verify tick count matches
        assertEquals(NUM_TICKS, harness.getTickCount(),
                "Harness should have processed exactly " + NUM_TICKS + " ticks");

        // 10. Print summary
        System.out.println("[Acceptance] === LEADERBOARD TOP 5 ===");
        for (int i = 0; i < Math.min(5, leaderboard.size()); i++) {
            Map<String, Object> e = leaderboard.get(i);
            System.out.printf("[Acceptance]   #%d %s | PnL=%.2f | Trades=%d | Ret=%.2f%% | Sharpe=%.4f%n",
                    e.get("rank"), e.get("agent_name"),
                    ((Number) e.get("total_pnl")).doubleValue(),
                    ((Number) e.get("trade_count")).intValue(),
                    ((Number) e.get("return_pct")).doubleValue(),
                    ((Number) e.get("sharpe_estimate")).doubleValue());
        }

        // 11. Verify all 24 codecs registered in CodecRegistry
        assertEquals(NUM_CODECS, CodecRegistry.registeredCount(),
                "CodecRegistry should have 24 registered codecs");
    }

    @Test
    @DisplayName("CLI-equivalent: --showdown --all-codecs --ticks 500 --simulated parsed and executed")
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void testShowdownViaCliPath() {
        // Parse CLI args same as Main.main() would
        String[] args = {"--showdown", "--all-codecs", "--ticks", "500", "--simulated"};
        ShowdownCli cli = ShowdownCli.parse(args);

        // Verify CLI parsing
        assertTrue(cli.isShowdown(), "Should be in showdown mode");
        assertEquals(NUM_CODECS, cli.getCodecIds().size(),
                "Should have 24 codec IDs from --all-codecs");
        assertEquals(NUM_TICKS, cli.getTicks(), "Should have 500 ticks");
        assertEquals(ShowdownCli.DataSourceKind.SIMULATED, cli.getDataSourceKind());

        // Build harness the same way Main.buildShowdownHarness() does for SIMULATED:
        // return new ShowdownHarness(cli.getCodecIds(), cli.getTicks());
        ShowdownHarness harness = new ShowdownHarness(cli.getCodecIds(), cli.getTicks());

        // Run
        long startTime = System.currentTimeMillis();
        Map<String, Map<String, Object>> results = harness.run();
        long elapsed = System.currentTimeMillis() - startTime;

        System.out.printf("[Acceptance-CLI] Showdown via CLI path completed in %d ms%n", elapsed);

        // Same validations as above
        assertTrue(elapsed < 60_000, "Should complete in < 60s, took: " + elapsed + " ms");
        assertEquals(NUM_CODECS, results.size(), "Should have 24 agent results");

        List<Map<String, Object>> leaderboard = harness.getLeaderboard();
        assertEquals(NUM_CODECS, leaderboard.size());

        // All agents must trade
        for (Map<String, Object> entry : leaderboard) {
            String name = (String) entry.get("agent_name");
            int tradeCount = ((Number) entry.get("trade_count")).intValue();
            assertTrue(tradeCount >= 1,
                    "Agent " + name + " must have >= 1 trade, got: " + tradeCount);
        }

        assertEquals(NUM_TICKS, harness.getTickCount());
    }

    @Test
    @DisplayName("Verify each of the 24 codecs produces trades independently")
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void testEachCodecProducesTradesIndependently() {
        // Test each codec individually to isolate any zero-trade issues
        List<String> zeroTradeCodecs = new ArrayList<>();

        for (int codecId = 1; codecId <= NUM_CODECS; codecId++) {
            List<Integer> ids = Collections.singletonList(codecId);
            ShowdownHarness harness = new ShowdownHarness(ids, NUM_TICKS);
            harness.run();

            Map<String, Map<String, Object>> summaries = harness.getSummary();
            assertEquals(1, summaries.size(), "Should have 1 agent result for codec " + codecId);

            Map<String, Object> summary = summaries.values().iterator().next();
            int tradeCount = ((Number) summary.get("trade_count")).intValue();

            if (tradeCount < 1) {
                String name = (String) summary.get("agent_name");
                zeroTradeCodecs.add(name + " (id=" + codecId + ")");
            }
        }

        assertTrue(zeroTradeCodecs.isEmpty(),
                "The following codecs produced zero trades: " + String.join(", ", zeroTradeCodecs));
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /**
     * Assert that all required summary metrics are present, have correct types,
     * and are finite (not NaN or Infinite).
     */
    private void assertValidMetrics(Map<String, Object> entry, String agentName) {
        String[] requiredFields = {
                "agent_name", "initial_cash", "final_value", "total_pnl",
                "return_pct", "realized_pnl", "unrealized_pnl",
                "sharpe_estimate", "hit_rate", "trade_count",
                "max_drawdown", "max_drawdown_pct", "ticks_processed"
        };

        for (String field : requiredFields) {
            assertTrue(entry.containsKey(field),
                    agentName + ": missing field '" + field + "'");
        }

        // Numeric validations
        assertFiniteNumber(entry.get("total_pnl"), agentName + ": total_pnl");
        assertFiniteNumber(entry.get("return_pct"), agentName + ": return_pct");
        assertFiniteNumber(entry.get("sharpe_estimate"), agentName + ": sharpe_estimate");
        assertFiniteNumber(entry.get("hit_rate"), agentName + ": hit_rate");
        assertFiniteNumber(entry.get("max_drawdown_pct"), agentName + ": max_drawdown_pct");
        assertFiniteNumber(entry.get("final_value"), agentName + ": final_value");

        // hit_rate should be in [0, 1]
        double hitRate = ((Number) entry.get("hit_rate")).doubleValue();
        assertTrue(hitRate >= 0.0 && hitRate <= 1.0,
                agentName + ": hit_rate should be in [0,1], got: " + hitRate);

        // max_drawdown_pct should be non-negative
        double maxDdPct = ((Number) entry.get("max_drawdown_pct")).doubleValue();
        assertTrue(maxDdPct >= 0.0,
                agentName + ": max_drawdown_pct should be >= 0, got: " + maxDdPct);

        // ticks_processed should match
        int ticksProcessed = ((Number) entry.get("ticks_processed")).intValue();
        assertEquals(NUM_TICKS, ticksProcessed,
                agentName + ": ticks_processed should be " + NUM_TICKS);
    }

    private void assertFiniteNumber(Object value, String context) {
        assertInstanceOf(Number.class, value, context + " should be a Number");
        double d = ((Number) value).doubleValue();
        assertTrue(Double.isFinite(d), context + " should be finite, got: " + d);
    }
}
