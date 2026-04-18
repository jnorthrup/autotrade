package com.xtrade.showdown;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the ShowdownCli argument parser and Main showdown integration.
 */
class ShowdownCliTest {

    // ── ShowdownCli.parse tests ──────────────────────────────────────────

    @Test
    void testNoArgsNotShowdown() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{});
        assertFalse(cli.isShowdown());
    }

    @Test
    void testDemoFlagNotShowdown() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--demo"});
        assertFalse(cli.isShowdown());
    }

    @Test
    void testShowdownFlag() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown"});
        assertTrue(cli.isShowdown());
    }

    @Test
    void testShowdownWithAllCodecs() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown", "--all-codecs"});
        assertTrue(cli.isShowdown());
        assertEquals(24, cli.getCodecIds().size());
        assertEquals(1, (int) cli.getCodecIds().get(0));
        assertEquals(24, (int) cli.getCodecIds().get(23));
    }

    @Test
    void testShowdownWithTicks() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown", "--all-codecs", "--ticks", "200"});
        assertTrue(cli.isShowdown());
        assertEquals(200, cli.getTicks());
    }

    @Test
    void testShowdownWithSimulated() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown", "--all-codecs", "--simulated"});
        assertTrue(cli.isShowdown());
        assertEquals(ShowdownCli.DataSourceKind.SIMULATED, cli.getDataSourceKind());
    }

    @Test
    void testShowdownWithReplay() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown", "--codecs", "1,2", "--replay", "/tmp/data.csv"});
        assertTrue(cli.isShowdown());
        assertEquals(ShowdownCli.DataSourceKind.REPLAY, cli.getDataSourceKind());
        assertEquals("/tmp/data.csv", cli.getReplayFile());
        assertEquals(Arrays.asList(1, 2), cli.getCodecIds());
    }

    @Test
    void testShowdownDefaultIsSimulated() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown"});
        assertEquals(ShowdownCli.DataSourceKind.SIMULATED, cli.getDataSourceKind());
    }

    @Test
    void testShowdownDefaultTicks() {
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown"});
        assertEquals(100, cli.getTicks());
    }

    @Test
    void testShowdownDefaultAllCodecs() {
        // Without --codecs or --all-codecs, showdown defaults to all 24
        ShowdownCli cli = ShowdownCli.parse(new String[]{"--showdown"});
        assertEquals(24, cli.getCodecIds().size());
    }

    @Test
    void testFullShowdownCommand() {
        ShowdownCli cli = ShowdownCli.parse(
                new String[]{"--showdown", "--all-codecs", "--ticks", "100", "--simulated"});
        assertTrue(cli.isShowdown());
        assertEquals(24, cli.getCodecIds().size());
        assertEquals(100, cli.getTicks());
        assertEquals(ShowdownCli.DataSourceKind.SIMULATED, cli.getDataSourceKind());
    }

    // ── expandCodecSpec tests ────────────────────────────────────────────

    @Test
    void testExpandSingleIds() {
        List<Integer> ids = ShowdownCli.expandCodecSpec("1,3,5");
        assertEquals(Arrays.asList(1, 3, 5), ids);
    }

    @Test
    void testExpandRange() {
        List<Integer> ids = ShowdownCli.expandCodecSpec("13-16");
        assertEquals(Arrays.asList(13, 14, 15, 16), ids);
    }

    @Test
    void testExpandMixed() {
        List<Integer> ids = ShowdownCli.expandCodecSpec("1,3,5,13-16");
        assertEquals(Arrays.asList(1, 3, 5, 13, 14, 15, 16), ids);
    }

    @Test
    void testExpandFullRange() {
        List<Integer> ids = ShowdownCli.expandCodecSpec("1-24");
        assertEquals(24, ids.size());
        assertEquals(1, (int) ids.get(0));
        assertEquals(24, (int) ids.get(23));
    }

    @Test
    void testExpandDeduplicates() {
        List<Integer> ids = ShowdownCli.expandCodecSpec("1,1,2,2,3");
        assertEquals(Arrays.asList(1, 2, 3), ids);
    }

    @Test
    void testExpandOutOfRangeThrows() {
        assertThrows(IllegalArgumentException.class, () -> ShowdownCli.expandCodecSpec("0"));
        assertThrows(IllegalArgumentException.class, () -> ShowdownCli.expandCodecSpec("25"));
        assertThrows(IllegalArgumentException.class, () -> ShowdownCli.expandCodecSpec("5-3"));
    }

    // ── Validation tests ─────────────────────────────────────────────────

    @Test
    void testBothAllCodecsAndCodecsThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> ShowdownCli.parse(new String[]{"--showdown", "--all-codecs", "--codecs", "1,2"}));
    }

    @Test
    void testTicksRequiresValue() {
        assertThrows(IllegalArgumentException.class,
                () -> ShowdownCli.parse(new String[]{"--showdown", "--ticks"}));
    }

    @Test
    void testReplayRequiresValue() {
        assertThrows(IllegalArgumentException.class,
                () -> ShowdownCli.parse(new String[]{"--showdown", "--replay"}));
    }

    @Test
    void testCodecsRequiresValue() {
        assertThrows(IllegalArgumentException.class,
                () -> ShowdownCli.parse(new String[]{"--showdown", "--codecs"}));
    }

    @Test
    void testTicksMustBePositive() {
        assertThrows(IllegalArgumentException.class,
                () -> ShowdownCli.parse(new String[]{"--showdown", "--ticks", "0"}));
    }

    @Test
    void testTicksNonIntegerThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> ShowdownCli.parse(new String[]{"--showdown", "--ticks", "abc"}));
    }

    // ── Main.runShowdown integration test ────────────────────────────────

    @Test
    void testMainRunShowdownWithSimulated() {
        // Build a CLI for a small showdown (3 agents, 10 ticks, simulated)
        ShowdownCli cli = ShowdownCli.parse(
                new String[]{"--showdown", "--codecs", "1,2,3", "--ticks", "10", "--simulated"});

        assertTrue(cli.isShowdown());
        assertEquals(Arrays.asList(1, 2, 3), cli.getCodecIds());
        assertEquals(10, cli.getTicks());
        assertEquals(ShowdownCli.DataSourceKind.SIMULATED, cli.getDataSourceKind());

        // Build and run a harness matching the CLI config
        ShowdownHarness harness = new ShowdownHarness(cli.getCodecIds(), cli.getTicks());
        var result = harness.run();

        assertNotNull(result);
        assertEquals(3, result.size());

        // Verify leaderboard
        List<java.util.Map<String, Object>> lb = harness.getLeaderboard();
        assertEquals(3, lb.size());
        // Verify ranks
        for (int i = 0; i < lb.size(); i++) {
            assertEquals(i + 1, ((Number) lb.get(i).get("rank")).intValue());
        }
    }

    @Test
    void testShowdownOutputFormat() {
        // Run a small showdown and capture the leaderboard output
        ShowdownHarness harness = new ShowdownHarness(Arrays.asList(1, 2), 10);
        harness.run();

        List<java.util.Map<String, Object>> lb = harness.getLeaderboard();
        for (java.util.Map<String, Object> entry : lb) {
            // Verify all required output columns exist
            assertTrue(entry.containsKey("rank"), "Missing rank");
            assertTrue(entry.containsKey("agent_name"), "Missing agent_name");
            assertTrue(entry.containsKey("total_pnl"), "Missing total_pnl (P&L)");
            assertTrue(entry.containsKey("return_pct"), "Missing return_pct");
            assertTrue(entry.containsKey("sharpe_estimate"), "Missing sharpe_estimate");
            assertTrue(entry.containsKey("hit_rate"), "Missing hit_rate");
            assertTrue(entry.containsKey("trade_count"), "Missing trade_count");
            assertTrue(entry.containsKey("max_drawdown_pct"), "Missing max_drawdown_pct");

            // Verify types are Numbers for formatting
            assertInstanceOf(Number.class, entry.get("rank"));
            assertInstanceOf(Number.class, entry.get("total_pnl"));
            assertInstanceOf(Number.class, entry.get("return_pct"));
            assertInstanceOf(Number.class, entry.get("sharpe_estimate"));
            assertInstanceOf(Number.class, entry.get("hit_rate"));
            assertInstanceOf(Number.class, entry.get("trade_count"));
            assertInstanceOf(Number.class, entry.get("max_drawdown_pct"));
        }
    }
}
