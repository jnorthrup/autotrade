package com.xtrade.codec;

import com.xtrade.kline.KlineBar;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Shared signal-preparation utilities used by showdown agents and live kline agents.
 */
public final class CodecSignalSupport {
    private CodecSignalSupport() {
    }

    public static Map<String, Object> marketDataFromBar(IndicatorComputer indicatorComputer, KlineBar bar) {
        Map<String, Object> marketData = new LinkedHashMap<>(indicatorComputer.compute(
                bar.seriesId().symbol(),
                bar.closePrice().doubleValue(),
                bar.baseVolume().doubleValue()));
        marketData.put("pair", bar.seriesId().symbol());
        marketData.put("exchange", bar.seriesId().venue());
        marketData.put("interval", bar.seriesId().interval().wireName());
        marketData.put("open", bar.openPrice().doubleValue());
        marketData.put("high", bar.highPrice().doubleValue());
        marketData.put("low", bar.lowPrice().doubleValue());
        marketData.put("price", bar.closePrice().doubleValue());
        marketData.put("close", bar.closePrice().doubleValue());
        marketData.put("volume", bar.baseVolume().doubleValue());
        marketData.put("quote_volume", bar.quoteVolume().doubleValue());
        marketData.put("closed", bar.closed());
        marketData.put("event_time", (double) bar.eventTimeMillis());
        return marketData;
    }

    public static Map<String, Double> extractInstrumentReadings(Map<String, Object> marketData, double price) {
        Map<String, Double> readings = new LinkedHashMap<>();
        readings.put("rsi", getDouble(marketData, "rsi", 50.0));
        readings.put("rsi_14", getDouble(marketData, "rsi_14", getDouble(marketData, "rsi", 50.0)));
        readings.put("macd_hist", getDouble(marketData, "macd_hist", 0.0));
        readings.put("atr_14", getDouble(marketData, "atr_14", 0.0));
        readings.put("bb_position", computeBbPosition(marketData, price));
        readings.put("adx", getDouble(marketData, "adx", getDouble(marketData, "adx_14", 0.0)));
        readings.put("adx_14", getDouble(marketData, "adx_14", getDouble(marketData, "adx", 0.0)));
        readings.put("vwap_ratio", computeVwapRatio(marketData, price));
        readings.put("momentum", getDouble(marketData, "momentum", 0.0));
        readings.put("macd", getDouble(marketData, "macd", 0.0));
        readings.put("macd_signal", getDouble(marketData, "macd_signal", 0.0));
        readings.put("plus_di", getDouble(marketData, "plus_di", 0.0));
        readings.put("minus_di", getDouble(marketData, "minus_di", 0.0));
        readings.put("stoch_k", getDouble(marketData, "stoch_k", 50.0));
        readings.put("stoch_d", getDouble(marketData, "stoch_d", 50.0));
        return readings;
    }

    public static double computeBbPosition(Map<String, Object> marketData, double price) {
        double bbUpper = getDouble(marketData, "bb_upper", price);
        double bbLower = getDouble(marketData, "bb_lower", price);
        double bbWidth = bbUpper - bbLower;
        if (bbWidth > 1e-12) {
            return (price - bbLower) / bbWidth;
        }
        return 0.5;
    }

    public static double computeVwapRatio(Map<String, Object> marketData, double price) {
        double vwap = getDouble(marketData, "vwap", price);
        return vwap != 0.0 ? price / vwap : 1.0;
    }

    public static double[] buildIndicatorVec(Map<String, Object> marketData) {
        double[] vec = new double[64];

        double price = getDouble(marketData, "price", 1.0);
        double safePrice = price != 0.0 ? price : 1.0;

        vec[0] = price / getDoubleNonZero(marketData, "sma_20", price);
        vec[1] = price / getDoubleNonZero(marketData, "sma_15", price);
        vec[2] = price / getDoubleNonZero(marketData, "ema_12", price);
        vec[3] = price / getDoubleNonZero(marketData, "ema_26", price);

        vec[4] = getDouble(marketData, "macd", 0.0) / safePrice;
        vec[5] = getDouble(marketData, "macd_signal", 0.0) / safePrice;
        vec[6] = getDouble(marketData, "macd_hist", 0.0) / safePrice;

        vec[7] = getDouble(marketData, "rsi", getDouble(marketData, "rsi_14", 50.0)) / 100.0;

        double bbUpper = getDouble(marketData, "bb_upper", price);
        double bbLower = getDouble(marketData, "bb_lower", price);
        double bbMid = getDouble(marketData, "bb_mid", price);
        double bbWidth = bbUpper - bbLower;
        if (bbWidth > 0) {
            vec[8] = (price - bbLower) / bbWidth;
        } else {
            vec[8] = 0.5;
        }
        vec[9] = bbMid != 0.0 ? bbWidth / bbMid : 0.0;

        vec[10] = getDouble(marketData, "atr_14", 0.0) / safePrice;
        vec[11] = getDouble(marketData, "stoch_k", 50.0) / 100.0;
        vec[12] = getDouble(marketData, "stoch_d", 50.0) / 100.0;
        vec[13] = getDouble(marketData, "adx", getDouble(marketData, "adx_14", 0.0)) / 100.0;
        vec[14] = getDouble(marketData, "plus_di", 0.0) / 100.0;
        vec[15] = getDouble(marketData, "minus_di", 0.0) / 100.0;

        double vwap = getDouble(marketData, "vwap", price);
        vec[16] = vwap != 0.0 ? price / vwap : 1.0;

        vec[17] = getDouble(marketData, "momentum", 0.0) / 100.0;

        double avgVol = getDouble(marketData, "avg_volume", 0.0);
        double vol = getDouble(marketData, "volume", 0.0);
        vec[18] = avgVol > 0.0 ? vol / avgVol : 1.0;

        vec[19] = getDouble(marketData, "log_return", 0.0);
        return vec;
    }

    private static double getDouble(Map<String, Object> marketData, String key, double defaultValue) {
        Object value = marketData.get(key);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return defaultValue;
    }

    private static double getDoubleNonZero(Map<String, Object> marketData, String key, double fallback) {
        double value = getDouble(marketData, key, fallback);
        return value != 0.0 ? value : fallback;
    }
}
