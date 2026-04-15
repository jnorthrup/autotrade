package com.xtrade;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Main application class, including bootstrap, trading loop,
 * signal execution, and portfolio summary.
 */
public class MainTest {

    @Test
    void testMainClassExists() {
        Main main = createDemoMain();
        assertNotNull(main);
    }

    @Test
    void testKrakenExchangeOnClasspath() throws Exception {
        Class<?> clazz = Class.forName("org.knowm.xchange.kraken.KrakenExchange");
        assertNotNull(clazz, "KrakenExchange class should be loadable");
        assertEquals("org.knowm.xchange.kraken.KrakenExchange", clazz.getName());
    }

    @Test
    void testMainCreatesWithDemoConfig() {
        Main app = Main.createApp(new String[]{"--demo"});
        assertNotNull(app);
        assertFalse(app.isRunning(), "Should not be running yet");
    }

    @Test
    void testMainFallsBackToDemoWithoutEnv() {
        // Without env vars set, it should fall back to demo mode
        Main app = Main.createApp(new String[]{});
        assertNotNull(app);
    }

    @Test
    void testTradingTickExecutesWithoutError() {
        Main app = createDemoMain();
        app.tradingTick();
        assertEquals(1, app.getTickCount());
    }

    @Test
    void testMultipleTicksExecute() {
        Main app = createDemoMain();
        for (int i = 0; i < 10; i++) {
            app.tradingTick();
        }
        assertEquals(10, app.getTickCount());
    }

    @Test
    void testPortfolioSummaryRunsAfterTicks() {
        Main app = createDemoMain();
        app.tradingTick();
        PortfolioSnapshot snapshot = app.getPaperEngine().getPortfolioSnapshot();
        assertNotNull(snapshot);
        assertTrue(snapshot.getTotalPortfolioValueUsd() > 0, "Starting balance should be > 0");
    }

    @Test
    void testSimulatedPricesGenerated() {
        Main app = createDemoMain();
        Map<TradingPair, ?> tickers = app.fetchPrices();
        assertNotNull(tickers);
        assertEquals(5, tickers.size(), "Should have 5 trading pairs");
        for (TradingPair pair : TradingPair.values()) {
            assertNotNull(tickers.get(pair), "Should have ticker for " + pair);
        }
    }

    @Test
    void testShutdownPersistsState() {
        Main app = createDemoMain();
        app.tradingTick();
        // Should not throw
        app.shutdown();
        assertFalse(app.isRunning(), "Should not be running after shutdown");
    }

    @Test
    void testSignalExecution() {
        Main app = createDemoMain();
        // Run enough ticks for strategy to potentially generate signals
        for (int i = 0; i < 15; i++) {
            app.tradingTick();
        }
        // Verify engine is consistent
        PortfolioSnapshot snapshot = app.getPaperEngine().getPortfolioSnapshot();
        assertNotNull(snapshot);
        assertTrue(snapshot.getTotalPortfolioValueUsd() > 0, "Portfolio value should remain positive");
    }

    @Test
    void testStrategyInitialized() {
        Main app = createDemoMain();
        TradingStrategy strategy = app.getStrategy();
        assertNotNull(strategy);
        assertEquals("SMA(3,7)", strategy.getName());
    }

    @Test
    void testRunFor10CyclesInThread() throws Exception {
        Main app = createDemoMainWithInterval(1);

        // Run the app in a background thread
        CountDownLatch started = new CountDownLatch(1);
        Thread runner = new Thread(() -> {
            started.countDown();
            app.run();
        }, "test-runner");
        runner.setDaemon(true);
        runner.start();

        assertTrue(started.await(2, TimeUnit.SECONDS), "App should start");

        // Wait for at least 10 ticks
        long deadline = System.currentTimeMillis() + 15_000;
        while (app.getTickCount() < 10 && System.currentTimeMillis() < deadline) {
            Thread.sleep(500);
        }

        assertTrue(app.getTickCount() >= 10,
                "Should have completed at least 10 ticks, got: " + app.getTickCount());

        app.shutdown();
        runner.join(5000);
    }

    // ------------------------------------------------------------------ //
    //  Helpers                                                            //
    // ------------------------------------------------------------------ //

    private Main createDemoMain() {
        return new Main(
                new AppConfig("demo-key", "demo-secret", "sandbox", new Properties()),
                null,
                true
        );
    }

    private Main createDemoMainWithInterval(int pollSeconds) {
        Properties props = new Properties();
        props.setProperty("poll-interval-seconds", String.valueOf(pollSeconds));
        return new Main(
                new AppConfig("demo-key", "demo-secret", "sandbox", props),
                null,
                true
        );
    }
}
