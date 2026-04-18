package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 06: Grid Trading
 *
 * Places orders at fixed grid intervals, profiting from oscillation.
 */
public class Codec06GridTrading extends BaseCodecExpert {

    public Codec06GridTrading() {
        super(6, "grid_trading", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double bbMid = getDouble(marketData, "bb_mid", price);
        double bbUpper = getDouble(marketData, "bb_upper", price);
        double bbLower = getDouble(marketData, "bb_lower", price);

        double direction = 0.0;
        double conviction = 0.3;

        double range = bbUpper - bbLower;
        if (range > 0) {
            double position = (price - bbLower) / range;
            if (position < 0.3) {
                direction = 1.0;
                conviction = 0.5 + (0.3 - position);
            } else if (position > 0.7) {
                direction = -1.0;
                conviction = 0.5 + (position - 0.7);
            }
        }

        recordInstruments("grid_position", range > 0 ? (price - bbLower) / range : 0.5);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
