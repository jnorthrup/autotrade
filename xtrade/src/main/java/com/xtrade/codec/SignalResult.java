package com.xtrade.codec;

/**
 * Result of a codec forward() call containing conviction and direction.
 * conviction ∈ [0, 1] — signal conviction strength
 * direction  ∈ [-1, 1] — negative = short, positive = long
 */
public class SignalResult {
    private final double conviction;
    private final double direction;

    public SignalResult(double conviction, double direction) {
        this.conviction = conviction;
        this.direction = direction;
    }

    public double getConviction() {
        return conviction;
    }

    public double getDirection() {
        return direction;
    }

    @Override
    public String toString() {
        return String.format("SignalResult{conviction=%.4f, direction=%.4f}", conviction, direction);
    }
}
