package com.xtrade.kline;

import java.util.List;

/**
 * Draw-thru cache interface for historical and streaming kline access.
 */
public interface DrawThruKlineFeed {
    KlineProducerHandle registerProducer(KlineProducerRegistration registration);

    List<KlineBar> requestBars(KlineBatchRequest request);

    KlineSubscription subscribe(KlineSubscriptionRequest request, KlineConsumer consumer);
}
