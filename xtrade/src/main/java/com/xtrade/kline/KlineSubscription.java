package com.xtrade.kline;

/**
 * Handle returned to a consumer subscription.
 */
public interface KlineSubscription extends AutoCloseable {
    String id();

    @Override
    void close();
}
