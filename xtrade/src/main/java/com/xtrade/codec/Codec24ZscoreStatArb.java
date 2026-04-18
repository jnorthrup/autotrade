package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 24: Z-Score Statistical Arbitrage
 *
 * Trades based on z-score of price deviation from statistical norms.
 */
public class Codec24ZscoreStatArb extends BaseCodecExpert {

    public Codec24ZscoreStatArb() {
        super(24, "zscore_stat_arb", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double sma20 = getDouble(marketData, "sma_20", price);
        double bbMid = getDouble(marketData, "bb_mid", price);
        double bbUpper = getDouble(marketData, "bb_upper", price);
        double bbLower = getDouble(marketData, "bb_lower", price);
        double rsi = getDouble(marketData, "rsi_14", 50.0);

        double bbWidth = bbUpper - bbLower;
        double zscore = 0.0;
        if (bbWidth > 0) {
            zscore = (price - bbMid) / (bbWidth / 4.0);
        }

        double direction = 0.0;
        double conviction = 0.0;

        if (Math.abs(zscore) > 1.5) {
            direction = zscore > 0 ? -1.0 : 1.0;
            conviction = Math.min(1.0, (Math.abs(zscore) - 1.5) / 2.0);
        } else if (Math.abs(zscore) > 1.0) {
            direction = zscore > 0 ? -1.0 : 1.0;
            conviction = Math.min(0.5, (Math.abs(zscore) - 1.0) / 2.0);
        }

        if ((direction > 0 && rsi < 35) || (direction < 0 && rsi > 65)) {
            conviction = Math.min(1.0, conviction * 1.3);
        }

        recordInstruments("zscore", zscore);
        recordInstruments("rsi_14", rsi);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
