package com.xtrade;

import com.xtrade.showdown.ShowdownCli;
import com.xtrade.showdown.ShowdownHarness;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Main entry point for the xtrade paper-trading application.
 * <p>
 * Bootstraps all components, runs a scheduled trading loop at the configured
 * poll interval, and handles graceful shutdown on SIGINT.
 * <p>
 * Supports two operating modes:
 * <ul>
 *   <li><b>Live</b> — connects to Kraken via XChange using real API credentials</li>
 *   <li><b>Demo</b> — uses simulated price data when credentials are unavailable</li>
 * </ul>
 * <p>
 * Run via: {@code mvn exec:java -Dexec.mainClass=com.xtrade.Main}
 */
public class Main {

    private static final Logger LOG = LoggerFactory.getLogger(Main.class);

    /** Default poll interval in seconds when not configured. */
    private static final int DEFAULT_POLL_INTERVAL = 5;

    /** Default initial virtual balance when not configured. */
    private static final double DEFAULT_BALANCE = 10000.00;

    /** Fixed trade size (USD) per signal execution. */
    private static final double TRADE_SIZE_USD = 100.00;

    // ---- Bootstrap state ----
    private final AppConfig appConfig;
    private final ExchangeService exchangeService;
    private final PaperTradingEngine paperEngine;
    private final TradingStrategy strategy;
    private final ScheduledExecutorService scheduler;
    private final int pollIntervalSeconds;
    private final boolean demoMode;
    private final AtomicInteger tickCount = new AtomicInteger(0);
    private final PortfolioReportPrinter reportPrinter = new PortfolioReportPrinter();

    /** Simulated prices for demo mode. */
    private final Map<TradingPair, Double> simulatedPrices = new EnumMap<>(TradingPair.class);
    private final Random random = new Random(42);

    /** Running flag. */
    private final AtomicBoolean running = new AtomicBoolean(false);

    // ------------------------------------------------------------------ //
    //                          Entry point                                // 
    // ------------------------------------------------------------------ //

    public static void main(String[] args) {
        // Parse CLI for showdown mode first
        ShowdownCli cli = ShowdownCli.parse(args);

        if (cli.isShowdown()) {
            runShowdown(cli);
            return;
        }

        printBanner();

        Main app = null;
        try {
            app = createApp(args);
        } catch (Exception e) {
            LOG.error("Failed to initialize application: {}", e.getMessage());
            System.err.println("[FATAL] Initialization failed: " + e.getMessage());
            System.exit(1);
        }

        app.run();
    }

    /**
     * Runs the multi-agent showdown mode.
     * Instantiates ShowdownHarness with the configured codec IDs,
     * data source, and tick count, then prints the leaderboard.
     */
    static void runShowdown(ShowdownCli cli) {
        System.out.println("========================================");
        System.out.println("   XT RADE  SHOWDOWN  MODE");
        System.out.println("========================================");
        System.out.printf("Codecs: %d agents | Ticks: %d | Source: %s | Output: %s%n",
                cli.getCodecIds().size(), cli.getTicks(), cli.getDataSourceKind(),
                cli.getOutputMode());
        if (cli.isDashboardEnabled()) {
            System.out.println("Dashboard: ENABLED");
        }

        ShowdownHarness harness = buildShowdownHarness(cli);

        // Apply output mode and dashboard settings
        ShowdownHarness.OutputMode harnessOutputMode =
                cli.getOutputMode() == ShowdownCli.OutputMode.JSON
                        ? ShowdownHarness.OutputMode.JSON
                        : ShowdownHarness.OutputMode.TEXT;
        harness.setOutputMode(harnessOutputMode, cli.getJsonOutputPath());
        harness.setDashboardEnabled(cli.isDashboardEnabled());

        System.out.println("Running showdown...");
        long startTime = System.currentTimeMillis();
        harness.run();
        long elapsed = System.currentTimeMillis() - startTime;

        // Leaderboard is printed inside harness.run() for TEXT mode,
        // JSON is written inside harness.run() for JSON mode.
        // Only print summary stats here if not already printed.
        if (cli.getOutputMode() == ShowdownCli.OutputMode.TEXT) {
            System.out.printf("Showdown completed in %d ms (%d ticks)%n", elapsed, harness.getTickCount());
        } else {
            System.out.printf("Showdown completed in %d ms (%d ticks)%n", elapsed, harness.getTickCount());
        }
    }

    /**
     * Constructs a ShowdownHarness from the parsed CLI configuration.
     */
    static ShowdownHarness buildShowdownHarness(ShowdownCli cli) {
        switch (cli.getDataSourceKind()) {
            case REPLAY:
                return new ShowdownHarness(
                        cli.getCodecIds(),
                        new com.xtrade.showdown.ReplayDataSource(cli.getReplayFile(), cli.getTicks()),
                        100_000.0,
                        java.util.Collections.singletonList("BTC/USDT"),
                        cli.getTicks()
                );
            case LIVE:
                // Attempt live; fall back to simulated if exchange is unavailable
                try {
                    AppConfig config = AppConfig.fromEnv();
                    ExchangeService svc = new ExchangeService(config);
                    return new ShowdownHarness(
                            cli.getCodecIds(),
                            new com.xtrade.showdown.RealtimeDataSource(
                                    java.util.Collections.singletonList("BTC/USD"),
                                    svc, cli.getTicks(), 1000L),
                            100_000.0,
                            java.util.Collections.singletonList("BTC/USD"),
                            cli.getTicks()
                    );
                } catch (Exception e) {
                    LOG.warn("Live data source unavailable, falling back to simulated: {}", e.getMessage());
                    System.out.println("[WARN] Live data unavailable, using simulated data source");
                    return new ShowdownHarness(cli.getCodecIds(), cli.getTicks());
                }
            case SIMULATED:
            default:
                return new ShowdownHarness(cli.getCodecIds(), cli.getTicks());
        }
    }

    // ------------------------------------------------------------------ //
    //                        Factory methods                              //
    // ------------------------------------------------------------------ //

    /**
     * Creates the application, attempting to load real config first and
     * falling back to demo mode if credentials are missing.
     */
    static Main createApp(String[] args) {
        boolean forceDemo = false;
        for (String arg : args) {
            if ("--demo".equalsIgnoreCase(arg)) {
                forceDemo = true;
            }
        }

        AppConfig config = null;
        ExchangeService exchangeService = null;
        boolean demoMode = forceDemo;

        if (!forceDemo) {
            try {
                config = AppConfig.fromEnv();
                exchangeService = new ExchangeService(config);
                LOG.info("Live mode: connected to Kraken exchange");
            } catch (Exception e) {
                LOG.warn("Cannot initialize live mode ({}), falling back to demo mode", e.getMessage());
                demoMode = true;
            }
        }

        if (demoMode) {
            Properties props = AppConfig.loadProperties();
            int pollInterval = AppConfig.parseInt(props, AppConfig.PROP_POLL_INTERVAL, DEFAULT_POLL_INTERVAL);
            BigDecimal balance = AppConfig.parseBigDecimal(props, AppConfig.PROP_INITIAL_VIRTUAL_BALANCE,
                    BigDecimal.valueOf(DEFAULT_BALANCE));

            config = new AppConfig("demo-key", "demo-secret", "sandbox", props);
            exchangeService = null;
            LOG.info("Demo mode: using simulated market data (poll interval={}s, balance=${})",
                    pollInterval, balance.toPlainString());
        }

        return new Main(config, exchangeService, demoMode);
    }

    // ------------------------------------------------------------------ //
    //                        Constructor                                  //
    // ------------------------------------------------------------------ //

    Main(AppConfig appConfig, ExchangeService exchangeService, boolean demoMode) {
        this.appConfig = appConfig;
        this.exchangeService = exchangeService;
        this.demoMode = demoMode;
        this.pollIntervalSeconds = Math.max(appConfig.getPollIntervalSeconds(), 1);
        this.paperEngine = new PaperTradingEngine(
                appConfig.getInitialVirtualBalance().doubleValue(), true);
        this.strategy = new SimpleMovingAverageStrategy();
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "trading-loop");
            t.setDaemon(true);
            return t;
        });

        // Initialize simulated prices for demo mode
        if (demoMode) {
            simulatedPrices.put(TradingPair.BTC_USD, 50000.0);
            simulatedPrices.put(TradingPair.ETH_USD, 3000.0);
            simulatedPrices.put(TradingPair.XRP_USD, 0.50);
            simulatedPrices.put(TradingPair.SOL_USD, 100.0);
            simulatedPrices.put(TradingPair.ADA_USD, 0.30);
        }
    }

    // ------------------------------------------------------------------ //
    //                        Application lifecycle                        //
    // ------------------------------------------------------------------ //

    /**
     * Starts the trading loop and blocks until shutdown.
     */
    void run() {
        if (!running.compareAndSet(false, true)) {
            LOG.warn("Application is already running");
            return;
        }

        Runtime.getRuntime().addShutdownHook(new Thread(this::shutdown, "shutdown-hook"));

        LOG.info("Starting trading loop (interval={}s, mode={})", pollIntervalSeconds,
                demoMode ? "DEMO" : "LIVE");
        System.out.printf("[%s] Trading loop starting | interval=%ds | mode=%s | balance=$%.2f%n",
                Instant.now(), pollIntervalSeconds, demoMode ? "DEMO" : "LIVE",
                appConfig.getInitialVirtualBalance());

        scheduler.scheduleAtFixedRate(this::tradingTick, 0, pollIntervalSeconds, TimeUnit.SECONDS);

        // Block main thread until shutdown
        try {
            while (running.get()) {
                Thread.sleep(1000);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Graceful shutdown: persists state, shuts down scheduler, logs final message.
     */
    void shutdown() {
        if (!running.compareAndSet(true, false)) {
            return;
        }

        String shutdownTime = Instant.now().toString();
        LOG.info("Shutdown initiated at {}", shutdownTime);
        System.out.printf("[%s] Shutting down...%n", shutdownTime);

        // Stop scheduler
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(10, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
                LOG.warn("Forced scheduler shutdown");
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }

        // Persist final portfolio state
        paperEngine.saveState();

        // Print final summary
        PortfolioSnapshot snapshot = paperEngine.getPortfolioSnapshot();
        printPortfolioSummary(snapshot);

        // Exchange cleanup (Exchange interface may not implement Closeable in XChange 4.4.2)
        if (exchangeService != null) {
            LOG.info("Exchange service released");
        }

        String msg = String.format(
                "[%s] Shutdown complete | ticks=%d | trades=%d | finalValue=$%.2f",
                Instant.now(), tickCount.get(), snapshot.getTradeHistory().size(),
                snapshot.getTotalPortfolioValueUsd());
        System.out.println(msg);
        LOG.info("Application shutdown complete");
    }

    // ------------------------------------------------------------------ //
    //                        Trading tick                                 //
    // ------------------------------------------------------------------ //

    /**
     * Executes one trading cycle: fetch prices, evaluate strategy,
     * execute signals, log actions, print summary.
     */
    void tradingTick() {
        try {
            int tick = tickCount.incrementAndGet();
            Instant now = Instant.now();
            System.out.printf("[%s] --- Tick #%d ---%n", now, tick);
            LOG.info("=== Tick #{} ===", tick);

            // Step 1: Fetch prices
            Map<TradingPair, Ticker> tickers = fetchPrices();
            if (tickers.isEmpty()) {
                LOG.warn("No price data available this tick");
                System.out.printf("[%s] WARNING: No price data available%n", Instant.now());
                return;
            }

            // Step 2: Update paper engine with current prices
            updateEnginePrices(tickers);

            // Step 3: Evaluate strategy for each pair and execute signals
            for (TradingPair pair : TradingPair.values()) {
                try {
                    Signal signal = strategy.evaluate(pair, tickers);
                    executeSignal(pair, signal, tickers);
                } catch (Exception e) {
                    LOG.error("Error processing {} on tick {}: {}", pair, tick, e.getMessage());
                    System.out.printf("[%s] ERROR processing %s: %s%n",
                            Instant.now(), pair, e.getMessage());
                }
            }

            // Step 4: Evaluate any open limit orders
            paperEngine.evaluateLimitOrders();

            // Step 5: Print portfolio summary
            PortfolioSnapshot snapshot = paperEngine.getPortfolioSnapshot();
            printPortfolioSummary(snapshot);

        } catch (Exception e) {
            LOG.error("Unhandled exception in trading tick: {}", e.getMessage(), e);
            System.out.printf("[%s] ERROR: %s%n", Instant.now(), e.getMessage());
        }
    }

    // ------------------------------------------------------------------ //
    //                        Price fetching                               //
    // ------------------------------------------------------------------ //

    /**
     * Fetches current ticker prices. In demo mode, generates simulated prices.
     */
    Map<TradingPair, Ticker> fetchPrices() {
        if (demoMode) {
            return generateSimulatedTickers();
        }

        try {
            return exchangeService.getAllTickers();
        } catch (Exception e) {
            LOG.error("Failed to fetch live prices: {}", e.getMessage());
            return Collections.emptyMap();
        }
    }

    /**
     * Generates simulated ticker data with random-walk price movements.
     */
    Map<TradingPair, Ticker> generateSimulatedTickers() {
        Map<TradingPair, Ticker> tickers = new EnumMap<>(TradingPair.class);

        for (TradingPair pair : TradingPair.values()) {
            double basePrice = simulatedPrices.getOrDefault(pair, 1.0);
            // Random walk: +/- 0.5% per tick
            double change = (random.nextDouble() - 0.48) * 0.01 * basePrice;
            double newPrice = Math.max(basePrice + change, 0.001);
            simulatedPrices.put(pair, newPrice);

            // Create a minimal Ticker-like object
            BigDecimal bdPrice = BigDecimal.valueOf(newPrice);
            Ticker ticker = new Ticker.Builder()
                    .currencyPair(pair.getCurrencyPair())
                    .last(bdPrice)
                    .bid(bdPrice.subtract(BigDecimal.valueOf(newPrice * 0.0001)))
                    .ask(bdPrice.add(BigDecimal.valueOf(newPrice * 0.0001)))
                    .build();
            tickers.put(pair, ticker);

            LOG.debug("[DEMO] {} simulated price: {}", pair, String.format("%.4f", newPrice));
        }

        return Collections.unmodifiableMap(tickers);
    }

    // ------------------------------------------------------------------ //
    //                     Price update helpers                            //
    // ------------------------------------------------------------------ //

    /**
     * Pushes current market prices into the paper trading engine.
     */
    void updateEnginePrices(Map<TradingPair, Ticker> tickers) {
        Map<String, Double> priceMap = new LinkedHashMap<>();
        for (Map.Entry<TradingPair, Ticker> entry : tickers.entrySet()) {
            if (entry.getValue().getLast() != null) {
                priceMap.put(entry.getKey().getSymbol(), entry.getValue().getLast().doubleValue());
            }
        }
        paperEngine.updateMarketPrices(priceMap);
    }

    // ------------------------------------------------------------------ //
    //                       Signal execution                              //
    // ------------------------------------------------------------------ //

    /**
     * Executes a trading signal if it is not HOLD.
     * Uses a fixed USD trade size to determine quantity.
     */
    void executeSignal(TradingPair pair, Signal signal, Map<TradingPair, Ticker> tickers) {
        if (signal == Signal.HOLD) {
            System.out.printf("[%s] %s -> HOLD%n", Instant.now(), pair);
            return;
        }

        Ticker ticker = tickers.get(pair);
        if (ticker == null || ticker.getLast() == null) {
            LOG.warn("Cannot execute signal for {}: no price data", pair);
            return;
        }

        double price = ticker.getLast().doubleValue();
        String symbol = pair.getSymbol();
        double quantity = TRADE_SIZE_USD / price;

        try {
            TradeRecord record;
            if (signal == Signal.BUY) {
                record = paperEngine.submitMarketOrder(symbol, "BUY", quantity);
                System.out.printf("[%s] EXECUTED BUY %s | qty=%.8f @ $%.2f | fee=$%.4f%n",
                        Instant.now(), pair, quantity, price,
                        record.getFee().doubleValue());
                LOG.info("Signal BUY executed for {}: {} @ {}", pair, String.format("%.8f", quantity),
                        String.format("%.2f", price));
            } else {
                // Only sell if we have a position
                String baseAsset = symbol.substring(0, symbol.indexOf('/'));
                double held = paperEngine.getHolding(baseAsset);
                if (held < quantity) {
                    quantity = held; // sell what we have
                }
                if (quantity <= 0) {
                    System.out.printf("[%s] %s -> SELL skipped (no position)%n", Instant.now(), pair);
                    LOG.debug("SELL signal for {} skipped: no position", pair);
                    return;
                }
                record = paperEngine.submitMarketOrder(symbol, "SELL", quantity);
                System.out.printf("[%s] EXECUTED SELL %s | qty=%.8f @ $%.2f | fee=$%.4f%n",
                        Instant.now(), pair, quantity, price,
                        record.getFee().doubleValue());
                LOG.info("Signal SELL executed for {}: {} @ {}", pair, String.format("%.8f", quantity),
                        String.format("%.2f", price));
            }
        } catch (Exception e) {
            LOG.error("Failed to execute {} signal for {}: {}", signal, pair, e.getMessage());
            System.out.printf("[%s] ERROR executing %s %s: %s%n",
                    Instant.now(), signal, pair, e.getMessage());
        }
    }

    // ------------------------------------------------------------------ //
    //                       Console output                                //
    // ------------------------------------------------------------------ //

    /**
     * Prints a formatted portfolio summary to console using PortfolioReportPrinter.
     */
    void printPortfolioSummary(PortfolioSnapshot snapshot) {
        reportPrinter.printReport(snapshot);
    }

    // ------------------------------------------------------------------ //
    //                       Banner                                        //
    // ------------------------------------------------------------------ //

    private static void printBanner() {
        String banner =
                "========================================\n"
                + "   _   _                _   \n"
                + "  | \\ | | _____   ____ | |_\n"
                + "  |  \\| |/ _ \\ \\ / / _ \\| __|\n"
                + "  | |\\  | (_) \\ V / (_) | |_\n"
                + "  |_| \\_|\\___/ \\_/ \\___/ \\__|\n"
                + "========================================";
        System.out.println(banner);
    }

    // ------------------------------------------------------------------ //
    //                   Package-private accessors (for testing)           //
    // ------------------------------------------------------------------ //

    PaperTradingEngine getPaperEngine() {
        return paperEngine;
    }

    TradingStrategy getStrategy() {
        return strategy;
    }

    int getTickCount() {
        return tickCount.get();
    }

    boolean isRunning() {
        return running.get();
    }
}
