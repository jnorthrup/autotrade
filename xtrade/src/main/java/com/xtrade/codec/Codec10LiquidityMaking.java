package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 10: Liquidity Making
 *
 * Market-making strategy profiting from bid-ask spread.
 */
public class Codec10LiquidityMaking extends BaseCodecExpert {

    public Codec10LiquidityMaking() {
        super(10, "liquidity_making", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double bbMid = getDouble(marketData, "bb_mid", price);
        double bbUpper = getDouble(marketData, "bb_upper", price);
        double bbLower = getDouble(marketData, "bb_lower", price);

        double direction = 0.0;
        double conviction = 0.2;

        if (price < bbMid) {
            direction = 1.0;
        } else {
            direction = -1.0;
        }

        conviction = Math.min(0.6, Math.abs(price - bbMid) / (bbMid > 0 ? bbMid : 1.0) * 10.0 + 0.2);

        recordInstruments("bb_mid", bbMid);
        recordInstruments("spread_estimate", (bbUpper - bbLower) / 2.0);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
