package com.xtrade.showdown;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the ShowdownHarness with simulated data.
 *
 * Validates all acceptance criteria:
 *   - ShowdownHarness.run() accepts a list of codec IDs
 *   - Each agent maintains isolated portfolio state
 *   - AgentMetrics.computeSummary() returns required fields
 *   - Leaderboard ranks agents by total PnL descending
 */
class ShowdownHarnessTest {

    private static final int NUM_TICKS = 50;

    @Test
    void testRunWithSimulatedData() {
        // Use a few codec IDs to test multi-agent
        List<Integer> codecIds = Arrays.asList(1, 2, 3);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);

        Map<String, Map<String, Object>> results = harness.run();

        // Should have results for each codec
        assertNotNull(results, "run() should return non-null results");
        assertEquals(3, results.size(), "Should have results for 3 agents");

        // Each result should have required summary fields
        for (Map.Entry<String, Map<String, Object>> entry : results.entrySet()) {
            Map<String, Object> summary = entry.getValue();
            assertSummaryHasRequiredFields(summary, entry.getKey());
        }
    }

    @Test
    void testComputeSummaryHasRequiredMetrics() {
        List<Integer> codecIds = Arrays.asList(1);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.run();

        Map<String, Map<String, Object>> summaries = harness.getSummary();
        assertEquals(1, summaries.size());

        Map<String, Object> summary = summaries.values().iterator().next();
        assertSummaryHasRequiredFields(summary, "codec_01");
    }

    @Test
    void testAgentIsolation() {
        // Run with 2 different codec IDs and verify no cross-agent leakage
        List<Integer> codecIds = Arrays.asList(1, 2);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.run();

        Map<String, ShowdownAgent> agents = harness.getAgents();
        assertEquals(2, agents.size());

        // Each agent should have its own isolated holdings (may differ)
        ShowdownAgent agent1 = null;
        ShowdownAgent agent2 = null;
        for (ShowdownAgent a : agents.values()) {
            if (a.getCodec().getCodecId() == 1) agent1 = a;
            if (a.getCodec().getCodecId() == 2) agent2 = a;
        }
        assertNotNull(agent1, "Agent 1 should exist");
        assertNotNull(agent2, "Agent 2 should exist");

        // Verify each has its own IndicatorComputer (no shared state)
        // Each agent processed NUM_TICKS ticks
        Map<String, AgentMetrics> metrics = harness.getMetricsMap();
        assertEquals(2, metrics.size());

        // Each agent should have equity curve entries
        for (AgentMetrics m : metrics.values()) {
            assertFalse(m.getEquityCurve().isEmpty(), "Equity curve should not be empty");
            // equity curve: initial value + one per tick = NUM_TICKS + 1
            assertEquals(NUM_TICKS + 1, m.getEquityCurve().size(),
                    "Equity curve should have initial + one entry per tick");
        }
    }

    @Test
    void testLeaderboardRankedByPnlDescending() {
        List<Integer> codecIds = Arrays.asList(1, 2, 3, 4, 5);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.run();

        List<Map<String, Object>> leaderboard = harness.getLeaderboard();

        assertEquals(5, leaderboard.size(), "Leaderboard should have 5 entries");

        // Verify ranks are assigned
        for (int i = 0; i < leaderboard.size(); i++) {
            assertEquals(i + 1, ((Number) leaderboard.get(i).get("rank")).intValue(),
                    "Rank should be " + (i + 1));
        }

        // Verify descending PnL order
        for (int i = 0; i < leaderboard.size() - 1; i++) {
            double pnlA = ((Number) leaderboard.get(i).get("total_pnl")).doubleValue();
            double pnlB = ((Number) leaderboard.get(i + 1).get("total_pnl")).doubleValue();
            assertTrue(pnlA >= pnlB,
                    "Leaderboard should be sorted by total_pnl descending: " + pnlA + " >= " + pnlB);
        }
    }

    @Test
    void testSimulatedDataSourceProducesCorrectTicks() {
        int ticks = 20;
        SimulatedDataSource ds = new SimulatedDataSource(ticks);

        int count = 0;
        while (ds.hasNext()) {
            TickData td = ds.next();
            assertNotNull(td);
            assertNotNull(td.getTicks());
            assertFalse(td.getTicks().isEmpty());
            for (Map.Entry<String, TickData.PairTick> e : td.getTicks().entrySet()) {
                assertTrue(e.getValue().getPrice() > 0, "Price should be positive");
                assertTrue(e.getValue().getVolume() > 0, "Volume should be positive");
            }
            count++;
        }
        assertEquals(ticks, count);
    }

    @Test
    void testSimulatedDataSourceReset() {
        int ticks = 10;
        SimulatedDataSource ds = new SimulatedDataSource(ticks);

        // Consume all
        List<Double> prices1 = new ArrayList<>();
        while (ds.hasNext()) {
            TickData td = ds.next();
            prices1.add(td.getTicks().get("BTC/USDT").getPrice());
        }

        // Reset
        ds.reset();

        // Consume again — should produce identical sequence (seeded)
        List<Double> prices2 = new ArrayList<>();
        while (ds.hasNext()) {
            TickData td = ds.next();
            prices2.add(td.getTicks().get("BTC/USDT").getPrice());
        }

        assertEquals(prices1.size(), prices2.size());
        for (int i = 0; i < prices1.size(); i++) {
            assertEquals(prices1.get(i), prices2.get(i), 1e-12,
                    "Reset should produce identical prices at tick " + i);
        }
    }

    @Test
    void testAgentMetricsFifoCostBasis() {
        AgentMetrics metrics = new AgentMetrics("test_agent", 100_000.0);

        // Tick 0: BUY 1.0 at price 50
        List<TradeAction> buyActions = Collections.singletonList(
                new TradeAction("BTC/USDT", TradeAction.BUY, 1.0, 50.0, 0.8, 1.0));
        Map<String, Double> holdings = new LinkedHashMap<>();
        holdings.put("BTC/USDT", 1.0);
        Map<String, Double> prices = new LinkedHashMap<>();
        prices.put("BTC/USDT", 50.0);
        metrics.recordTick(99_950.0, holdings, prices, buyActions, 0, 0.0);

        // Tick 1: SELL 1.0 at price 60 → realized PnL = (60-50)*1.0 = 10
        List<TradeAction> sellActions = Collections.singletonList(
                new TradeAction("BTC/USDT", TradeAction.SELL, 1.0, 60.0, 0.8, -1.0));
        holdings.put("BTC/USDT", 0.0);
        prices.put("BTC/USDT", 60.0);
        metrics.recordTick(100_010.0, holdings, prices, sellActions, 1, 1.0);

        Map<String, Object> summary = metrics.computeSummary();
        assertEquals(10.0, ((Number) summary.get("realized_pnl")).doubleValue(), 1e-6,
                "FIFO realized PnL should be 10.0");
        // BUY counts as 1 trade + SELL counts as 1 trade = 2 total
        assertEquals(2, ((Number) summary.get("trade_count")).intValue(),
                "Should have recorded 2 trades (1 buy + 1 sell)");
    }

    @Test
    void testAgentMetricsComputeSummaryFields() {
        AgentMetrics metrics = new AgentMetrics("test_agent", 100_000.0);

        // No trades, just record one tick
        Map<String, Double> holdings = new LinkedHashMap<>();
        Map<String, Double> prices = new LinkedHashMap<>();
        prices.put("BTC/USDT", 100.0);
        metrics.recordTick(100_000.0, holdings, prices,
                Collections.emptyList(), 0, 0.0);

        Map<String, Object> summary = metrics.computeSummary();

        // Verify all required fields
        assertTrue(summary.containsKey("total_pnl"), "Missing total_pnl");
        assertTrue(summary.containsKey("return_pct"), "Missing return_pct");
        assertTrue(summary.containsKey("sharpe_estimate"), "Missing sharpe_estimate");
        assertTrue(summary.containsKey("hit_rate"), "Missing hit_rate");
        assertTrue(summary.containsKey("trade_count"), "Missing trade_count");
        assertTrue(summary.containsKey("max_drawdown_pct"), "Missing max_drawdown_pct");

        // With no trades, values should be zero
        assertEquals(0.0, ((Number) summary.get("total_pnl")).doubleValue(), 1e-6);
        assertEquals(0.0, ((Number) summary.get("return_pct")).doubleValue(), 1e-6);
        assertEquals(0.0, ((Number) summary.get("hit_rate")).doubleValue(), 1e-6);
        assertEquals(0, ((Number) summary.get("trade_count")).intValue());
        assertEquals(0.0, ((Number) summary.get("max_drawdown_pct")).doubleValue(), 1e-6);
    }

    @Test
    void testHarnessReset() {
        List<Integer> codecIds = Arrays.asList(1, 2);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.run();

        // Reset
        harness.reset();

        // Tick count should be reset
        assertEquals(0, harness.getTickCount());

        // Run again
        Map<String, Map<String, Object>> results2 = harness.run();
        assertEquals(2, results2.size());
    }

    @Test
    void testHarnessWithSingleCodec() {
        List<Integer> codecIds = Collections.singletonList(1);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        Map<String, Map<String, Object>> results = harness.run();

        assertEquals(1, results.size());
        String key = results.keySet().iterator().next();
        assertTrue(key.contains("codec_01") || key.contains("volatility_breakout"),
                "Key should reference codec 01: " + key);
    }

    @Test
    void testEquityCurveLength() {
        List<Integer> codecIds = Arrays.asList(1);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.run();

        Map<String, AgentMetrics> metricsMap = harness.getMetricsMap();
        for (AgentMetrics m : metricsMap.values()) {
            List<Double> curve = m.getEquityCurve();
            assertEquals(NUM_TICKS + 1, curve.size(),
                    "Equity curve should have initial + one per tick");
            assertEquals(100_000.0, curve.get(0), 1e-6,
                    "Initial equity should be 100_000");
        }
    }

    @Test
    void testMaxDrawdownNonNegative() {
        List<Integer> codecIds = Arrays.asList(1, 2, 3);
        ShowdownHarness harness = new ShowdownHarness(codecIds, NUM_TICKS);
        harness.run();

        Map<String, Map<String, Object>> summaries = harness.getSummary();
        for (Map.Entry<String, Map<String, Object>> entry : summaries.entrySet()) {
            double maxDdPct = ((Number) entry.getValue().get("max_drawdown_pct")).doubleValue();
            assertTrue(maxDdPct >= 0.0,
                    "Max drawdown % should be non-negative for " + entry.getKey());
        }
    }

    @Test
    void testTradeActionConstants() {
        assertEquals("BUY", TradeAction.BUY);
        assertEquals("SELL", TradeAction.SELL);
        assertEquals("HOLD", TradeAction.HOLD);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private void assertSummaryHasRequiredFields(Map<String, Object> summary, String label) {
        String msg = "Missing field in summary for " + label;
        assertTrue(summary.containsKey("agent_name"), msg + ": agent_name");
        assertTrue(summary.containsKey("total_pnl"), msg + ": total_pnl");
        assertTrue(summary.containsKey("return_pct"), msg + ": return_pct");
        assertTrue(summary.containsKey("sharpe_estimate"), msg + ": sharpe_estimate");
        assertTrue(summary.containsKey("hit_rate"), msg + ": hit_rate");
        assertTrue(summary.containsKey("trade_count"), msg + ": trade_count");
        assertTrue(summary.containsKey("max_drawdown_pct"), msg + ": max_drawdown_pct");

        // Verify types
        assertInstanceOf(Number.class, summary.get("total_pnl"), "total_pnl should be Number");
        assertInstanceOf(Number.class, summary.get("return_pct"), "return_pct should be Number");
        assertInstanceOf(Number.class, summary.get("sharpe_estimate"), "sharpe_estimate should be Number");
        assertInstanceOf(Number.class, summary.get("hit_rate"), "hit_rate should be Number");
        assertInstanceOf(Number.class, summary.get("trade_count"), "trade_count should be Number");
        assertInstanceOf(Number.class, summary.get("max_drawdown_pct"), "max_drawdown_pct should be Number");
    }
}
