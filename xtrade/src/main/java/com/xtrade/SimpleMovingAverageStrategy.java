package com.xtrade;

import org.knowm.xchange.dto.marketdata.Ticker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.Objects;

/**
 * Simple moving-average crossover strategy.
 * <p>
 * Maintains a sliding window of the last {@code longWindow} closing prices.
 * Generates a {@link Signal#BUY} when the short-period SMA crosses above the
 * long-period SMA, and a {@link Signal#SELL} when it crosses below.
 * Returns {@link Signal#HOLD} when there is insufficient data or no crossover.
 * <p>
 * Thread-safe: each trading pair has its own independent price history.
 */
public class SimpleMovingAverageStrategy implements TradingStrategy {

    private static final Logger LOG = LoggerFactory.getLogger(SimpleMovingAverageStrategy.class);

    /** Default short moving-average window size. */
    static final int DEFAULT_SHORT_WINDOW = 3;

    /** Default long moving-average window size. */
    static final int DEFAULT_LONG_WINDOW = 7;

    private final int shortWindow;
    private final int longWindow;
    private final Map<TradingPair, PriceWindow> histories;

    /**
     * Creates a strategy with default window sizes (short=3, long=7).
     */
    public SimpleMovingAverageStrategy() {
        this(DEFAULT_SHORT_WINDOW, DEFAULT_LONG_WINDOW);
    }

    /**
     * Creates a strategy with custom window sizes.
     *
     * @param shortWindow short SMA period (must be >= 2)
     * @param longWindow  long SMA period (must be > shortWindow)
     */
    public SimpleMovingAverageStrategy(int shortWindow, int longWindow) {
        if (shortWindow < 2) {
            throw new IllegalArgumentException("shortWindow must be >= 2, got: " + shortWindow);
        }
        if (longWindow <= shortWindow) {
            throw new IllegalArgumentException(
                    "longWindow must be > shortWindow, got short=" + shortWindow + " long=" + longWindow);
        }
        this.shortWindow = shortWindow;
        this.longWindow = longWindow;

        java.util.Map<TradingPair, PriceWindow> h = new java.util.EnumMap<>(TradingPair.class);
        for (TradingPair tp : TradingPair.values()) {
            h.put(tp, new PriceWindow(longWindow));
        }
        this.histories = java.util.Collections.unmodifiableMap(h);
    }

    @Override
    public String getName() {
        return "SMA(" + shortWindow + "," + longWindow + ")";
    }

    @Override
    public Signal evaluate(TradingPair pair, Map<TradingPair, Ticker> tickers) {
        Objects.requireNonNull(pair, "pair must not be null");
        Objects.requireNonNull(tickers, "tickers must not be null");

        Ticker ticker = tickers.get(pair);
        if (ticker == null || ticker.getLast() == null) {
            LOG.debug("No ticker data for {}, returning HOLD", pair);
            return Signal.HOLD;
        }

        double price = ticker.getLast().doubleValue();
        PriceWindow window = histories.get(pair);
        if (window == null) {
            return Signal.HOLD;
        }

        int prevSize = window.size();
        window.add(price);

        // Need at least longWindow + 1 prices to detect a crossover
        if (window.size() < longWindow + 1) {
            LOG.debug("{}: collecting data ({}/{}), HOLD", pair, window.size(), longWindow + 1);
            return Signal.HOLD;
        }

        // Current short and long SMAs
        double currentShortSma = window.sma(shortWindow);
        double currentLongSma = window.sma(longWindow);

        // Previous short and long SMAs (exclude the most recent price)
        double prevShortSma = window.sma(shortWindow, 1);
        double prevLongSma = window.sma(longWindow, 1);

        Signal signal;
        // Golden cross: short crosses above long
        if (prevShortSma <= prevLongSma && currentShortSma > currentLongSma) {
            signal = Signal.BUY;
        }
        // Death cross: short crosses below long
        else if (prevShortSma >= prevLongSma && currentShortSma < currentLongSma) {
            signal = Signal.SELL;
        } else {
            signal = Signal.HOLD;
        }

        LOG.info("[STRATEGY] {} | price={:.2f} | shortSMA={:.2f} | longSMA={:.2f} | signal={}",
                pair, price, currentShortSma, currentLongSma, signal);

        return signal;
    }

    @Override
    public void reset() {
        for (PriceWindow window : histories.values()) {
            window.clear();
        }
        LOG.info("Strategy {} reset", getName());
    }

    /**
     * Returns the current price history size for a pair (for testing).
     */
    int getHistorySize(TradingPair pair) {
        PriceWindow window = histories.get(pair);
        return window != null ? window.size() : 0;
    }

    // ------------------------------------------------------------------ //
    //  Internal sliding-window helper                                     //
    // ------------------------------------------------------------------ //

    /**
     * Fixed-size sliding window of double prices.
     * Calculates simple moving averages over any period up to the capacity.
     */
    static class PriceWindow {
        private final int capacity;
        private final Deque<Double> prices;

        PriceWindow(int capacity) {
            this.capacity = capacity;
            this.prices = new ArrayDeque<>(capacity + 1);
        }

        /** Adds a price, evicting the oldest if over capacity. */
        void add(double price) {
            prices.addLast(price);
            while (prices.size() > capacity + 1) {
                prices.removeFirst();
            }
        }

        /** Number of prices stored. */
        int size() {
            return prices.size();
        }

        /** Clears all stored prices. */
        void clear() {
            prices.clear();
        }

        /**
         * Computes the SMA of the last {@code period} prices, starting from
         * the most recent element.
         *
         * @param period number of elements to average
         * @return the simple moving average
         */
        double sma(int period) {
            return sma(period, 0);
        }

        /**
         * Computes the SMA of the last {@code period} prices, offset back
         * by {@code offset} elements from the most recent.
         *
         * @param period number of elements to average
         * @param offset elements to skip from the tail
         * @return the simple moving average
         */
        double sma(int period, int offset) {
            if (period <= 0 || period + offset > prices.size()) {
                return Double.NaN;
            }
            Double[] arr = prices.toArray(new Double[0]);
            double sum = 0.0;
            int end = arr.length - 1 - offset;
            int start = end - period + 1;
            for (int i = start; i <= end; i++) {
                sum += arr[i];
            }
            return sum / period;
        }
    }
}
