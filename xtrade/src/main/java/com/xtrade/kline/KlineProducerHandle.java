package com.xtrade.kline;

import java.util.Collection;

/**
 * Producer handle returned after registration.
 */
public interface KlineProducerHandle {
    KlineProducerRegistration registration();

    void publish(KlineBar bar);

    void publishAll(Collection<KlineBar> bars);
}
