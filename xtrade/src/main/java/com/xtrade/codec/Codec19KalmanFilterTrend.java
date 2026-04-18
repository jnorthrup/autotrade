package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 19: Kalman Filter Trend
 *
 * Uses a simple Kalman-style state estimator for trend detection.
 */
public class Codec19KalmanFilterTrend extends BaseCodecExpert {

    private double estimate = 0.0;
    private double errorCov = 1.0;
    private boolean initialized = false;

    public Codec19KalmanFilterTrend() {
        super(19, "kalman_filter_trend", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double price = getDouble(marketData, "price", 0.0);
        double atr14 = getDouble(marketData, "atr_14", 1.0);

        if (!initialized) {
            estimate = price;
            initialized = true;
        }

        double q = 0.01;
        double r = atr14 > 0 ? atr14 * atr14 * 0.01 : 1.0;

        errorCov += q;
        double kalmanGain = errorCov / (errorCov + r);
        double prevEstimate = estimate;
        estimate = estimate + kalmanGain * (price - estimate);
        errorCov = (1.0 - kalmanGain) * errorCov;

        double trend = estimate - prevEstimate;
        double direction = trend > 0 ? 1.0 : -1.0;
        double conviction = Math.min(1.0, Math.abs(trend) / (atr14 > 0 ? atr14 : 1.0));

        recordInstruments("kalman_estimate", estimate);
        recordInstruments("kalman_trend", trend);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }

    @Override
    public void reset() {
        estimate = 0.0;
        errorCov = 1.0;
        initialized = false;
    }
}
