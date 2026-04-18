package com.xtrade.codec;

import java.util.Map;

/**
 * Codec 23: Transformer Attention
 *
 * Attention-mechanism inspired weighting of indicator features.
 */
public class Codec23TransformerAttention extends BaseCodecExpert {

    public Codec23TransformerAttention() {
        super(23, "transformer_attention", "1.0");
    }

    @Override
    public SignalResult forward(Map<String, Object> marketData, double[] indicatorVec) {
        double rsi = getDouble(marketData, "rsi_14", 50.0);
        double macdHist = getDouble(marketData, "macd_hist", 0.0);
        double momentum = getDouble(marketData, "momentum", 0.0);
        double adx = getDouble(marketData, "adx_14", 0.0);
        double stochK = getDouble(marketData, "stoch_k", 50.0);
        double price = getDouble(marketData, "price", 0.0);

        double rsiFeat = (rsi - 50.0) / 50.0;
        double macdFeat = macdHist * 100.0;
        double momFeat = momentum / 5.0;
        double stochFeat = (stochK - 50.0) / 50.0;
        double adxFeat = adx / 50.0;

        double[] features = {rsiFeat, macdFeat, momFeat, stochFeat, adxFeat};
        double[] weights = {0.3, 0.25, 0.2, 0.15, 0.1};

        double maxAbs = 0.0;
        for (double f : features) maxAbs = Math.max(maxAbs, Math.abs(f));
        double temperature = maxAbs > 0 ? maxAbs : 1.0;

        double weightedSum = 0.0;
        double weightSum = 0.0;
        for (int i = 0; i < features.length; i++) {
            double attention = Math.exp(features[i] / temperature);
            weightedSum += features[i] * weights[i] * attention;
            weightSum += weights[i] * attention;
        }

        double signal = weightSum > 0 ? weightedSum / weightSum : 0.0;
        double direction = signal > 0 ? 1.0 : -1.0;
        double conviction = Math.min(1.0, Math.abs(signal));

        recordInstruments("attention_signal", signal);
        recordInstruments("temperature", temperature);

        updateObMemory(direction, padOrTruncate(indicatorVec));
        return validateSignal(conviction, direction);
    }
}
