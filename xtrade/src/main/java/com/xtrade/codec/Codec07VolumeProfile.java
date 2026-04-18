package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 07: Volume Profile
 *
 * Analyzes volume distribution to identify support/resistance.
 */
public class Codec07VolumeProfile extends BaseCodecExpert {

    public Codec07VolumeProfile() {
        super(7, "volume_profile", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double volume = getDouble(marketData, "volume", 0.0);
        double avgVolume = getDouble(marketData, "avg_volume", 1.0);
        double vwap = getDouble(marketData, "vwap", price);

        double direction = 0.0;
        double conviction = 0.0;

        double volumeRatio = avgVolume > 0 ? volume / avgVolume : 1.0;
        if (volumeRatio > 1.5) {
            conviction = Math.min(1.0, volumeRatio / 3.0);
            direction = price > vwap ? 1.0 : -1.0;
        }

        recordInstruments("volume_ratio", volumeRatio);
        recordInstruments("vwap_deviation", vwap > 0 ? (price - vwap) / vwap : 0.0);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
