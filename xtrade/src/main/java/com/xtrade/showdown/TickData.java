package com.xtrade.showdown;

import java.util.Map;

/**
 * Represents one tick of market data: a mapping of pair -> (price, volume).
 * All agents receive the same TickData each cycle.
 */
public class TickData {

    private final Map<String, PairTick> ticks;

    public TickData(Map<String, PairTick> ticks) {
        this.ticks = ticks;
    }

    public Map<String, PairTick> getTicks() {
        return ticks;
    }

    /**
     * Price and volume for a single pair at one point in time.
     */
    public static class PairTick {
        private final double price;
        private final double volume;

        public PairTick(double price, double volume) {
            this.price = price;
            this.volume = volume;
        }

        public double getPrice() {
            return price;
        }

        public double getVolume() {
            return volume;
        }
    }
}
