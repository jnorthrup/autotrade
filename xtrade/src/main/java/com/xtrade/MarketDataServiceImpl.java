package com.xtrade;

import org.knowm.xchange.Exchange;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.service.marketdata.MarketDataService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Wraps XChange's MarketDataService to fetch real-time ticker data for a fixed
 * set of Kraken crypto pairs (BTC/USD, ETH/USD, XRP/USD, SOL/USD, ADA/USD).
 * <p>
 * Provides a polling mechanism driven by a {@link ScheduledExecutorService} that
 * retrieves snapshots at a configurable interval and logs key metrics at INFO level.
 * Any API call failure is caught, logged at ERROR level, and does not crash the application.
 */
public class MarketDataServiceImpl {

    private static final Logger LOG = LoggerFactory.getLogger(MarketDataServiceImpl.class);

    /** Default polling interval in seconds. */
    static final long DEFAULT_POLL_INTERVAL_SECONDS = 60L;

    /** The 5 major Kraken crypto pairs to monitor. */
    static final List<CurrencyPair> MONITORED_PAIRS;
    static {
        List<CurrencyPair> pairs = new ArrayList<>();
        pairs.add(CurrencyPair.BTC_USD);
        pairs.add(CurrencyPair.ETH_USD);
        pairs.add(CurrencyPair.XRP_USD);
        pairs.add(new CurrencyPair("SOL/USD"));  // SOL_USD not a pre-defined constant in XChange 5.2.0
        pairs.add(CurrencyPair.ADA_USD);
        MONITORED_PAIRS = Collections.unmodifiableList(pairs);
    }

    private final MarketDataService marketDataService;
    private final long pollIntervalSeconds;
    private final ScheduledExecutorService scheduler;
    private final AtomicBoolean running = new AtomicBoolean(false);

    /** Most recent ticker snapshot keyed by CurrencyPair. */
    private volatile Map<CurrencyPair, Ticker> latestTickers = Collections.emptyMap();

    /**
     * Creates a MarketDataServiceImpl with the default polling interval (60 seconds).
     *
     * @param exchange the XChange Exchange instance (must not be null)
     */
    public MarketDataServiceImpl(Exchange exchange) {
        this(exchange, DEFAULT_POLL_INTERVAL_SECONDS);
    }

    /**
     * Creates a MarketDataServiceImpl with a configurable polling interval.
     *
     * @param exchange             the XChange Exchange instance (must not be null)
     * @param pollIntervalSeconds  polling interval in seconds (must be &gt; 0)
     */
    public MarketDataServiceImpl(Exchange exchange, long pollIntervalSeconds) {
        Objects.requireNonNull(exchange, "Exchange must not be null");
        if (pollIntervalSeconds <= 0) {
            throw new IllegalArgumentException("Poll interval must be > 0, got: " + pollIntervalSeconds);
        }
        this.marketDataService = exchange.getMarketDataService();
        this.pollIntervalSeconds = pollIntervalSeconds;
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "market-data-poller");
            t.setDaemon(true);
            return t;
        });
        LOG.info("MarketDataServiceImpl initialized with {} pairs, poll interval={}s",
                MONITORED_PAIRS.size(), pollIntervalSeconds);
    }

    // ---- one-shot fetch ----

    /**
     * Fetches tickers for all monitored pairs in a single pass.
     * Failures for individual pairs are logged at ERROR level and do not prevent
     * the remaining pairs from being fetched.
     *
     * @return an unmodifiable map of CurrencyPair -&gt; Ticker for all successfully fetched pairs
     */
    public Map<CurrencyPair, Ticker> fetchAllTickers() {
        Map<CurrencyPair, Ticker> results = new LinkedHashMap<>();
        for (CurrencyPair pair : MONITORED_PAIRS) {
            try {
                Ticker ticker = marketDataService.getTicker(pair);
                if (ticker != null) {
                    results.put(pair, ticker);
                    logTicker(pair, ticker);
                }
            } catch (IOException e) {
                LOG.error("Failed to fetch ticker for {}: {}", pair, e.getMessage(), e);
            } catch (RuntimeException e) {
                LOG.error("Unexpected error fetching ticker for {}: {}", pair, e.getMessage(), e);
            }
        }
        this.latestTickers = Collections.unmodifiableMap(results);
        return latestTickers;
    }

    // ---- polling ----

    /**
     * Starts the polling loop. If already running, this is a no-op.
     * The first fetch happens immediately, then repeatedly at the configured interval.
     */
    public void startPolling() {
        if (running.compareAndSet(false, true)) {
            LOG.info("Starting market data polling every {}s", pollIntervalSeconds);
            scheduler.scheduleAtFixedRate(() -> {
                try {
                    fetchAllTickers();
                } catch (Exception e) {
                    // Belt-and-suspenders: catch anything to prevent scheduler death
                    LOG.error("Unhandled exception in polling cycle: {}", e.getMessage(), e);
                }
            }, 0, pollIntervalSeconds, TimeUnit.SECONDS);
        } else {
            LOG.warn("Polling is already running; ignoring duplicate start request");
        }
    }

    /**
     * Gracefully stops the polling loop and shuts down the executor.
     */
    public void stopPolling() {
        if (running.compareAndSet(true, false)) {
            LOG.info("Stopping market data polling");
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(10, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                    LOG.warn("Forced shutdown of market data scheduler");
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Returns whether the polling loop is currently active.
     */
    public boolean isRunning() {
        return running.get();
    }

    /**
     * Returns the most recently fetched ticker snapshots.
     */
    public Map<CurrencyPair, Ticker> getLatestTickers() {
        return latestTickers;
    }

    /**
     * Returns the configured polling interval in seconds.
     */
    public long getPollIntervalSeconds() {
        return pollIntervalSeconds;
    }

    /**
     * Returns the monitored currency pairs.
     */
    public List<CurrencyPair> getMonitoredPairs() {
        return MONITORED_PAIRS;
    }

    // ---- internal ----

    /**
     * Logs a single ticker at INFO level with symbol, last price, bid, ask, and 24h volume.
     */
    void logTicker(CurrencyPair pair, Ticker ticker) {
        BigDecimal last = ticker.getLast();
        BigDecimal bid = ticker.getBid();
        BigDecimal ask = ticker.getAsk();
        BigDecimal volume = ticker.getVolume();
        LOG.info("[MARKET] {} | last={} | bid={} | ask={} | 24hVol={}",
                pair,
                last != null ? last.toPlainString() : "N/A",
                bid != null ? bid.toPlainString() : "N/A",
                ask != null ? ask.toPlainString() : "N/A",
                volume != null ? volume.toPlainString() : "N/A");
    }
}
