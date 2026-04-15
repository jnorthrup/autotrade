package com.xtrade;

import org.knowm.xchange.Exchange;
import org.knowm.xchange.ExchangeFactory;
import org.knowm.xchange.ExchangeSpecification;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.dto.marketdata.OrderBook;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.dto.trade.LimitOrder;
import org.knowm.xchange.kraken.KrakenExchange;
import org.knowm.xchange.service.marketdata.MarketDataService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Core exchange service that connects to Kraken via XChange using credentials
 * from {@link AppConfig}. Provides methods to fetch current ticker prices and
 * order-book depth for each of the 5 supported trading pairs.
 * <p>
 * Includes rate-limit awareness (minimum delay between API calls) and basic
 * retry logic with exponential backoff for transient network errors.
 * <p>
 * All calls use Kraken's public endpoints — no trading/authentication required
 * beyond initial exchange construction.
 */
public class ExchangeService {

    private static final Logger LOG = LoggerFactory.getLogger(ExchangeService.class);

    /** Maximum number of retry attempts for transient errors. */
    static final int DEFAULT_MAX_RETRIES = 3;

    /** Initial delay in milliseconds before first retry. */
    static final long DEFAULT_INITIAL_RETRY_DELAY_MS = 1000L;

    /** Minimum gap in milliseconds between consecutive API calls (rate-limit). */
    static final long DEFAULT_RATE_LIMIT_GAP_MS = 500L;

    private final Exchange exchange;
    private final MarketDataService marketDataService;
    private final int maxRetries;
    private final long initialRetryDelayMs;
    private final long rateLimitGapMs;

    /** Timestamp of the last API call, used to enforce rate-limiting. */
    private volatile long lastCallTimestampMs = 0L;

    /**
     * Creates an ExchangeService by building a KrakenExchange with credentials
     * from the given {@link AppConfig}. Uses default retry and rate-limit settings.
     *
     * @param appConfig application configuration containing Kraken API credentials
     * @throws NullPointerException if appConfig is null
     */
    public ExchangeService(AppConfig appConfig) {
        this(appConfig, DEFAULT_MAX_RETRIES, DEFAULT_INITIAL_RETRY_DELAY_MS, DEFAULT_RATE_LIMIT_GAP_MS);
    }

    /**
     * Creates an ExchangeService with custom retry and rate-limit parameters.
     *
     * @param appConfig            application configuration with Kraken credentials
     * @param maxRetries           maximum number of retry attempts (must be >= 0)
     * @param initialRetryDelayMs  initial delay in ms before first retry (must be > 0)
     * @param rateLimitGapMs       minimum gap in ms between consecutive API calls (must be >= 0)
     */
    ExchangeService(AppConfig appConfig, int maxRetries, long initialRetryDelayMs, long rateLimitGapMs) {
        Objects.requireNonNull(appConfig, "AppConfig must not be null");
        if (maxRetries < 0) {
            throw new IllegalArgumentException("maxRetries must be >= 0, got: " + maxRetries);
        }
        if (initialRetryDelayMs <= 0) {
            throw new IllegalArgumentException("initialRetryDelayMs must be > 0, got: " + initialRetryDelayMs);
        }

        this.maxRetries = maxRetries;
        this.initialRetryDelayMs = initialRetryDelayMs;
        this.rateLimitGapMs = rateLimitGapMs;

        ExchangeSpecification spec = new KrakenExchange().getDefaultExchangeSpecification();
        spec.setApiKey(appConfig.getApiKey());
        spec.setSecretKey(appConfig.getApiSecret());

        this.exchange = ExchangeFactory.INSTANCE.createExchange(spec);
        this.marketDataService = exchange.getMarketDataService();

        LOG.info("ExchangeService initialized with KrakenExchange (maxRetries={}, initialRetryDelay={}ms, rateLimitGap={}ms)",
                maxRetries, initialRetryDelayMs, rateLimitGapMs);
    }

    /**
     * Creates an ExchangeService from a pre-built Exchange instance.
     * Package-private — used for testing with mocked exchanges.
     */
    ExchangeService(Exchange exchange, int maxRetries, long initialRetryDelayMs, long rateLimitGapMs) {
        Objects.requireNonNull(exchange, "Exchange must not be null");
        this.exchange = exchange;
        this.marketDataService = exchange.getMarketDataService();
        this.maxRetries = maxRetries;
        this.initialRetryDelayMs = initialRetryDelayMs;
        this.rateLimitGapMs = rateLimitGapMs;
    }

    // ------------------------------------------------------------------ //
    //                        Public API                                   //
    // ------------------------------------------------------------------ //

    /**
     * Fetches the current ticker for the given currency pair.
     * Returns a {@link Ticker} containing last price, bid, ask, and volume.
     * <p>
     * Implements retry with exponential backoff for transient network errors.
     *
     * @param pair the currency pair to query (must not be null)
     * @return a Ticker with last price, bid, ask, and volume
     * @throws IOException if all retry attempts fail
     * @throws NullPointerException if pair is null
     */
    public Ticker getCurrentTicker(CurrencyPair pair) throws IOException {
        Objects.requireNonNull(pair, "CurrencyPair must not be null");
        return executeWithRetry(() -> {
            enforceRateLimit();
            Ticker ticker = marketDataService.getTicker(pair);
            if (ticker == null) {
                throw new IOException("Received null ticker for " + pair);
            }
            logTicker(pair, ticker);
            return ticker;
        }, "getCurrentTicker(" + pair + ")");
    }

    /**
     * Fetches the current order book for the given currency pair.
     * Returns bids and asks up to the specified depth.
     * <p>
     * Implements retry with exponential backoff for transient network errors.
     *
     * @param pair  the currency pair to query (must not be null)
     * @param depth maximum number of levels to return for bids and asks
     *              (pass 0 for exchange default depth)
     * @return an {@link OrderBook} containing bids and asks
     * @throws IOException if all retry attempts fail
     * @throws NullPointerException if pair is null
     */
    public OrderBook getOrderBook(CurrencyPair pair, int depth) throws IOException {
        Objects.requireNonNull(pair, "CurrencyPair must not be null");
        return executeWithRetry(() -> {
            enforceRateLimit();
            OrderBook orderBook;
            if (depth > 0) {
                orderBook = marketDataService.getOrderBook(pair, depth);
            } else {
                orderBook = marketDataService.getOrderBook(pair);
            }
            if (orderBook == null) {
                throw new IOException("Received null order book for " + pair);
            }
            LOG.debug("[ORDERBOOK] {} | bids={} | asks={}", pair,
                    orderBook.getBids().size(), orderBook.getAsks().size());
            return orderBook;
        }, "getOrderBook(" + pair + ", depth=" + depth + ")");
    }

    /**
     * Fetches tickers for all 5 supported trading pairs.
     * Individual pair failures are logged but do not prevent other pairs from being fetched.
     *
     * @return an unmodifiable map of CurrencyPair to Ticker for successfully fetched pairs
     */
    public Map<TradingPair, Ticker> getAllTickers() {
        Map<TradingPair, Ticker> results = new EnumMap<>(TradingPair.class);
        for (TradingPair tp : TradingPair.values()) {
            try {
                Ticker ticker = getCurrentTicker(tp.getCurrencyPair());
                results.put(tp, ticker);
            } catch (IOException e) {
                LOG.error("Failed to fetch ticker for {} after retries: {}", tp, e.getMessage());
            }
        }
        return Collections.unmodifiableMap(results);
    }

    /**
     * Fetches order books for all 5 supported trading pairs.
     * Individual pair failures are logged but do not prevent other pairs from being fetched.
     *
     * @param depth maximum number of levels per side (0 for exchange default)
     * @return an unmodifiable map of TradingPair to OrderBook for successfully fetched pairs
     */
    public Map<TradingPair, OrderBook> getAllOrderBooks(int depth) {
        Map<TradingPair, OrderBook> results = new EnumMap<>(TradingPair.class);
        for (TradingPair tp : TradingPair.values()) {
            try {
                OrderBook ob = getOrderBook(tp.getCurrencyPair(), depth);
                results.put(tp, ob);
            } catch (IOException e) {
                LOG.error("Failed to fetch order book for {} after retries: {}", tp, e.getMessage());
            }
        }
        return Collections.unmodifiableMap(results);
    }

    // ------------------------------------------------------------------ //
    //                     Accessors / helpers                             //
    // ------------------------------------------------------------------ //

    /** Returns the underlying XChange Exchange instance. */
    public Exchange getExchange() {
        return exchange;
    }

    /** Returns the configured maximum number of retries. */
    public int getMaxRetries() {
        return maxRetries;
    }

    /** Returns the configured rate-limit gap in milliseconds. */
    public long getRateLimitGapMs() {
        return rateLimitGapMs;
    }

    /**
     * Returns the list of supported trading pairs from AppConfig.
     */
    public List<TradingPair> getSupportedPairs() {
        return TradingPair.valuesAsList();
    }

    // ------------------------------------------------------------------ //
    //                     Internal implementation                         //
    // ------------------------------------------------------------------ //

    /**
     * Functional interface for retryable operations.
     */
    @FunctionalInterface
    interface RetryableOperation<T> {
        T execute() throws IOException;
    }

    /**
     * Executes an operation with retry logic using exponential backoff.
     * Retries on IOException (transient network errors). RuntimeExceptions
     * are not retried (they indicate programming errors).
     *
     * @param operation the operation to execute
     * @param description human-readable description for logging
     * @param <T> return type
     * @return the result of the operation
     * @throws IOException if all attempts fail
     */
    <T> T executeWithRetry(RetryableOperation<T> operation, String description) throws IOException {
        IOException lastException = null;

        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return operation.execute();
            } catch (IOException e) {
                lastException = e;
                if (attempt < maxRetries) {
                    long delayMs = initialRetryDelayMs * (1L << attempt); // exponential backoff
                    LOG.warn("[RETRY] {} failed (attempt {}/{}): {}. Retrying in {}ms...",
                            description, attempt + 1, maxRetries + 1, e.getMessage(), delayMs);
                    sleepQuietly(delayMs);
                } else {
                    LOG.error("[RETRY] {} failed after {} attempts: {}", description, maxRetries + 1, e.getMessage());
                }
            }
        }

        throw lastException;
    }

    /**
     * Enforces rate-limiting by sleeping if the last API call was too recent.
     */
    void enforceRateLimit() {
        if (rateLimitGapMs <= 0) return;

        long now = System.currentTimeMillis();
        long elapsed = now - lastCallTimestampMs;
        if (elapsed < rateLimitGapMs) {
            long sleepMs = rateLimitGapMs - elapsed;
            LOG.trace("[RATE-LIMIT] Sleeping {}ms to respect rate limit", sleepMs);
            sleepQuietly(sleepMs);
        }
        lastCallTimestampMs = System.currentTimeMillis();
    }

    /**
     * Sleeps for the specified duration without throwing checked exceptions.
     * InterruptedException is restored on the thread's interrupt flag.
     */
    static void sleepQuietly(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Logs a ticker at INFO level.
     */
    void logTicker(CurrencyPair pair, Ticker ticker) {
        BigDecimal last = ticker.getLast();
        BigDecimal bid = ticker.getBid();
        BigDecimal ask = ticker.getAsk();
        BigDecimal volume = ticker.getVolume();
        LOG.info("[TICKER] {} | last={} | bid={} | ask={} | vol={}",
                pair,
                last != null ? last.toPlainString() : "N/A",
                bid != null ? bid.toPlainString() : "N/A",
                ask != null ? ask.toPlainString() : "N/A",
                volume != null ? volume.toPlainString() : "N/A");
    }

    /**
     * Validates that all 5 supported trading pairs can return a ticker
     * without throwing an exception. Useful for smoke-testing connectivity.
     *
     * @return true if all 5 pairs returned a non-null ticker
     */
    public boolean validateAllPairs() {
        boolean allOk = true;
        for (TradingPair tp : TradingPair.values()) {
            try {
                Ticker ticker = getCurrentTicker(tp.getCurrencyPair());
                if (ticker == null || ticker.getLast() == null) {
                    LOG.warn("Ticker for {} returned null or missing last price", tp);
                    allOk = false;
                }
            } catch (IOException e) {
                LOG.error("Validation failed for {}: {}", tp, e.getMessage());
                allOk = false;
            }
        }
        return allOk;
    }
}
