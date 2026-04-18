package com.xtrade.showdown;

import com.xtrade.ExchangeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Real-time data source that polls via ExchangeService.
 *
 * On each tick, fetches current ticker prices from the exchange.
 * Falls back to simulated random walk if exchange is unavailable.
 */
public class RealtimeDataSource implements IDataSource {

    private static final Logger LOG = LoggerFactory.getLogger(RealtimeDataSource.class);

    private final List<String> pairs;
    private final ExchangeService exchangeService;
    private final Integer maxTicks;
    private final long pollIntervalMs;

    private int tickIdx;
    private final Random fallbackRng = new Random(99);
    private final Map<String, Double> fallbackPrices;

    /**
     * @param pairs           trading pairs, e.g. ["BTC/USD"]
     * @param exchangeService the ExchangeService to poll
     * @param maxTicks        maximum ticks (null = unlimited)
     * @param pollIntervalMs  milliseconds between polls
     */
    public RealtimeDataSource(
            List<String> pairs,
            ExchangeService exchangeService,
            Integer maxTicks,
            long pollIntervalMs
    ) {
        this.pairs = pairs != null ? new ArrayList<>(pairs) : Collections.singletonList("BTC/USD");
        this.exchangeService = exchangeService;
        this.maxTicks = maxTicks;
        this.pollIntervalMs = pollIntervalMs;
        this.tickIdx = 0;
        this.fallbackPrices = new LinkedHashMap<>();
        for (String p : this.pairs) {
            fallbackPrices.put(p, 100.0);
        }
    }

    public RealtimeDataSource(ExchangeService exchangeService, int maxTicks) {
        this(Collections.singletonList("BTC/USD"), exchangeService, maxTicks, 1000L);
    }

    @Override
    public boolean hasNext() {
        if (maxTicks != null && tickIdx >= maxTicks) return false;
        return true; // realtime never ends (bounded by maxTicks)
    }

    @Override
    public TickData next() {
        if (!hasNext()) throw new NoSuchElementException("No more ticks");

        TickData data = pollExchange();
        tickIdx++;

        if (pollIntervalMs > 0) {
            try {
                Thread.sleep(pollIntervalMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        return data;
    }

    private TickData pollExchange() {
        if (exchangeService == null) return fallback();

        try {
            var tickers = exchangeService.getAllTickers();
            Map<String, TickData.PairTick> result = new LinkedHashMap<>();
            for (String pair : pairs) {
                // Try to find matching ticker
                boolean found = false;
                for (var entry : tickers.entrySet()) {
                    String tpStr = entry.getKey().toString();
                    if (tpStr.equals(pair) || pair.startsWith(tpStr.split("/")[0])) {
                        var ticker = entry.getValue();
                        double price = ticker.getLast() != null ? ticker.getLast().doubleValue() : 100.0;
                        double vol = ticker.getVolume() != null ? ticker.getVolume().doubleValue() : 0.0;
                        result.put(pair, new TickData.PairTick(price, vol));
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    result.put(pair, fallbackEntry(pair));
                }
            }
            return new TickData(result);
        } catch (Exception e) {
            LOG.warn("Exchange poll failed, using fallback: {}", e.getMessage());
            return fallback();
        }
    }

    private TickData fallback() {
        Map<String, TickData.PairTick> result = new LinkedHashMap<>();
        for (String pair : pairs) {
            result.put(pair, fallbackEntry(pair));
        }
        return new TickData(result);
    }

    private TickData.PairTick fallbackEntry(String pair) {
        double lr = 0.005 * fallbackRng.nextGaussian();
        double old = fallbackPrices.get(pair);
        double newPrice = old * Math.exp(lr);
        fallbackPrices.put(pair, newPrice);
        double vol = 100.0 + fallbackRng.nextDouble() * 4900.0;
        return new TickData.PairTick(newPrice, vol);
    }

    @Override
    public void reset() {
        tickIdx = 0;
        for (String p : pairs) {
            fallbackPrices.put(p, 100.0);
        }
    }
}
