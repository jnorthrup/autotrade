package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 11: Sector Rotation
 *
 * Rotates between sectors based on relative strength.
 */
public class Codec11SectorRotation extends BaseCodecExpert {

    public Codec11SectorRotation() {
        super(11, "sector_rotation", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double sma20 = getDouble(marketData, "sma_20", price);
        double momentum = getDouble(marketData, "momentum", 0.0);
        double adx = getDouble(marketData, "adx_14", 0.0);

        double direction = momentum > 0 ? 1.0 : -1.0;
        double conviction = adx > 20 ? Math.min(1.0, Math.abs(momentum) / 5.0) : 0.1;

        if (price < sma20 && momentum > 0) {
            conviction = Math.min(1.0, conviction + 0.3);
        } else if (price > sma20 && momentum < 0) {
            conviction = Math.min(1.0, conviction + 0.3);
        }

        recordInstruments("momentum", momentum);
        recordInstruments("adx", adx);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
