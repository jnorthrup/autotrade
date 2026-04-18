package com.xtrade.showdown;

import java.util.*;

/**
 * Simulated random-walk data source.
 * Generates synthetic tick data for one or more trading pairs.
 *
 * Mirrors Python SimulatedDataSource: each tick, log-returns are drawn
 * from a normal distribution and prices evolve as geometric Brownian motion.
 */
public class SimulatedDataSource implements IDataSource {

    private final List<String> pairs;
    private final int numTicks;
    private final Map<String, Double> basePrices;
    private final long seed;
    private final double drift;
    private final double volatility;

    private Random rng;
    private int tickIdx;
    private Map<String, Double> prices;

    /**
     * @param pairs       trading pairs, e.g. ["BTC/USDT"]
     * @param numTicks    total ticks to produce
     * @param basePrices  starting price per pair (may be null, defaults to 100.0)
     * @param seed        random seed
     * @param drift       mean log-return per tick
     * @param volatility  std-dev of log-return per tick
     */
    public SimulatedDataSource(
            List<String> pairs,
            int numTicks,
            Map<String, Double> basePrices,
            long seed,
            double drift,
            double volatility
    ) {
        this.pairs = pairs != null ? new ArrayList<>(pairs) : Collections.singletonList("BTC/USDT");
        this.numTicks = numTicks;
        this.basePrices = basePrices != null ? new LinkedHashMap<>(basePrices)
                : new LinkedHashMap<>();
        for (String p : this.pairs) {
            this.basePrices.putIfAbsent(p, 100.0);
        }
        this.seed = seed;
        this.drift = drift;
        this.volatility = volatility;

        this.rng = new Random(seed);
        this.tickIdx = 0;
        this.prices = new LinkedHashMap<>(this.basePrices);
    }

    public SimulatedDataSource(List<String> pairs, int numTicks) {
        this(pairs, numTicks, null, 42L, 0.0001, 0.02);
    }

    public SimulatedDataSource(int numTicks) {
        this(Collections.singletonList("BTC/USDT"), numTicks, null, 42L, 0.0001, 0.02);
    }

    @Override
    public boolean hasNext() {
        return tickIdx < numTicks;
    }

    @Override
    public TickData next() {
        if (!hasNext()) throw new NoSuchElementException("No more ticks");

        Map<String, TickData.PairTick> ticks = new LinkedHashMap<>();
        for (String pair : pairs) {
            double logRet = drift + volatility * rng.nextGaussian();
            double oldPrice = prices.get(pair);
            double newPrice = oldPrice * Math.exp(logRet);
            prices.put(pair, newPrice);
            double vol = 100.0 + rng.nextDouble() * 9900.0; // uniform [100, 10000]
            ticks.put(pair, new TickData.PairTick(newPrice, vol));
        }
        tickIdx++;
        return new TickData(ticks);
    }

    @Override
    public void reset() {
        this.rng = new Random(seed);
        this.tickIdx = 0;
        this.prices = new LinkedHashMap<>(basePrices);
    }
}
