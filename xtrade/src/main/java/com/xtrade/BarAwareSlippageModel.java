package com.xtrade;

import com.xtrade.kline.KlineBar;

/**
 * Slippage model that applies fixed basis points plus a simple bar-volume participation impact.
 */
public final class BarAwareSlippageModel implements SlippageModel {
    private final double basisPoints;
    private final double volumeImpactMultiplier;

    public BarAwareSlippageModel(double basisPoints, double volumeImpactMultiplier) {
        if (basisPoints < 0.0) {
            throw new IllegalArgumentException("basisPoints must be >= 0");
        }
        if (volumeImpactMultiplier < 0.0) {
            throw new IllegalArgumentException("volumeImpactMultiplier must be >= 0");
        }
        this.basisPoints = basisPoints;
        this.volumeImpactMultiplier = volumeImpactMultiplier;
    }

    @Override
    public double fillPrice(TradeRecord.Side side, double quantity, KlineBar bar) {
        double close = bar.closePrice().doubleValue();
        double high = bar.highPrice().doubleValue();
        double low = bar.lowPrice().doubleValue();
        double baseVolume = Math.max(bar.baseVolume().doubleValue(), 1e-12);
        double participation = Math.max(quantity, 0.0) / baseVolume;
        double impact = basisPoints + (participation * volumeImpactMultiplier);
        if (side == TradeRecord.Side.BUY) {
            return clamp(close * (1.0 + impact), low, high);
        }
        return clamp(close * (1.0 - impact), low, high);
    }

    public double getBasisPoints() {
        return basisPoints;
    }

    public double getVolumeImpactMultiplier() {
        return volumeImpactMultiplier;
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
}
