package com.xtrade;

import com.xtrade.kline.KlineBar;

/**
 * Computes a simulated fill price for paper orders executed against a kline bar.
 */
public interface SlippageModel {
    double fillPrice(TradeRecord.Side side, double quantity, KlineBar bar);

    static SlippageModel none() {
        return (side, quantity, bar) -> bar.closePrice().doubleValue();
    }
}
