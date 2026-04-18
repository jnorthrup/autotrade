package com.xtrade.kline.binance;

import java.time.Clock;

public final class BinanceRateLimiter {
    public interface Sleeper {
        void sleep(long millis) throws InterruptedException;
    }

    private final long minIntervalMillis;
    private final Clock clock;
    private final Sleeper sleeper;
    private long lastAcquireMillis = Long.MIN_VALUE;

    public BinanceRateLimiter(long minIntervalMillis) {
        this(minIntervalMillis, Clock.systemUTC(), Thread::sleep);
    }

    BinanceRateLimiter(long minIntervalMillis, Clock clock, Sleeper sleeper) {
        if (minIntervalMillis < 0) {
            throw new IllegalArgumentException("minIntervalMillis must be >= 0");
        }
        this.minIntervalMillis = minIntervalMillis;
        this.clock = clock;
        this.sleeper = sleeper;
    }

    public synchronized void acquire() throws InterruptedException {
        if (lastAcquireMillis == Long.MIN_VALUE) {
            lastAcquireMillis = clock.millis();
            return;
        }
        long now = clock.millis();
        long waitMillis = minIntervalMillis - (now - lastAcquireMillis);
        if (waitMillis > 0) {
            sleeper.sleep(waitMillis);
            now = clock.millis();
        }
        lastAcquireMillis = now;
    }
}
