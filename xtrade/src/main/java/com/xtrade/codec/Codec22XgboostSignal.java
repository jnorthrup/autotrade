package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 22: XGBoost Signal
 *
 * Gradient-boosted decision tree inspired signal with feature weights.
 */
public class Codec22XgboostSignal extends BaseCodecExpert {

    public Codec22XgboostSignal() {
        super(22, "xgboost_signal", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double rsi = getDouble(marketData, "rsi_14", 50.0);
        double macdHist = getDouble(marketData, "macd_hist", 0.0);
        double adx = getDouble(marketData, "adx_14", 0.0);
        double momentum = getDouble(marketData, "momentum", 0.0);
        double stochK = getDouble(marketData, "stoch_k", 50.0);
        double atr14 = getDouble(marketData, "atr_14", 0.0);
        double price = getDouble(marketData, "price", 0.0);

        double score = 0.0;
        score += ((rsi - 50.0) / 50.0) * 0.25;
        score += (macdHist > 0 ? 1.0 : -1.0) * 0.20;
        score += (momentum > 0 ? 1.0 : -1.0) * 0.20;
        score += ((stochK - 50.0) / 50.0) * 0.15;
        score += (adx > 25 ? 0.3 : -0.1) * 0.20;

        double direction = score > 0 ? 1.0 : -1.0;
        double conviction = Math.min(1.0, Math.abs(score) * 2.0);

        recordInstruments("xgb_score", score);
        recordInstruments("rsi_14", rsi);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
