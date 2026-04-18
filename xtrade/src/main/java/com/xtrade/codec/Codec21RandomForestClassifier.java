package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 21: Random Forest Classifier
 *
 * Ensemble decision-tree inspired signal aggregation.
 */
public class Codec21RandomForestClassifier extends BaseCodecExpert {

    public Codec21RandomForestClassifier() {
        super(21, "random_forest_classifier", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double rsi = getDouble(marketData, "rsi_14", 50.0);
        double macdHist = getDouble(marketData, "macd_hist", 0.0);
        double adx = getDouble(marketData, "adx_14", 0.0);
        double stochK = getDouble(marketData, "stoch_k", 50.0);
        double momentum = getDouble(marketData, "momentum", 0.0);
        double bbMid = getDouble(marketData, "bb_mid", 0.0);
        double price = getDouble(marketData, "price", 0.0);

        int votes = 0;
        if (rsi < 40) votes++;
        else if (rsi > 60) votes--;
        if (macdHist > 0) votes++;
        else votes--;
        if (momentum > 0) votes++;
        else votes--;
        if (stochK < 50) votes++;
        else votes--;
        if (adx > 25) {
            votes += (macdHist > 0) ? 1 : -1;
        }

        double direction = votes > 0 ? 1.0 : (votes < 0 ? -1.0 : 0.0);
        double conviction = Math.min(1.0, Math.abs(votes) / 5.0);

        recordInstruments("ensemble_votes", votes);
        recordInstruments("rsi_14", rsi);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
