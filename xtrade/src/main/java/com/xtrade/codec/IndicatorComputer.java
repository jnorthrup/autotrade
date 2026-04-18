package com.xtrade.codec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Java port of showdown/indicators.py IndicatorComputer.
 *
 * Maintains per-pair rolling OHLCV buffers and computes all required
 * technical indicators on each tick. All computation is pure Java
 * (no external TA library) mirroring the pure-numpy Python logic.
 *
 * Implemented indicators:
 *   SMA(15), SMA(20)
 *   EMA(12), EMA(26)
 *   MACD line / signal / histogram (12/26/9)
 *   RSI(14)
 *   Bollinger Bands(20, 2)
 *   ATR(14)
 *   Stochastic %K(14) / %D(3)
 *   ADX(14) with +DI / -DI
 *   VWAP (rolling cumulative)
 *   Momentum (10-bar rate of change)
 */
public class IndicatorComputer {

    private final int bufferSize;
    private final Map<String, BarBuffer> buffers;
    private final ReentrantLock lock;

    public IndicatorComputer() {
        this(200);
    }

    public IndicatorComputer(int bufferSize) {
        this.bufferSize = bufferSize;
        this.buffers = new ConcurrentHashMap<>();
        this.lock = new ReentrantLock();
    }

    // ── Public API ────────────────────────────────────────────────────

    /**
     * Record a new tick and return a market_data dict with all indicators.
     */
    public Map<String, Object> compute(String pair, double price, double volume) {
        lock.lock();
        try {
            BarBuffer buf = buffers.computeIfAbsent(pair, k -> new BarBuffer(bufferSize));
            buf.appendTick(price, volume);
            return computeIndicators(buf, price, volume);
        } finally {
            lock.unlock();
        }
    }

    // ── SMA ───────────────────────────────────────────────────────────

    private static double sma(double[] arr, int period) {
        int n = arr.length;
        if (n < period) {
            return n > 0 ? arr[n - 1] : 0.0;
        }
        double sum = 0.0;
        for (int i = n - period; i < n; i++) {
            sum += arr[i];
        }
        return sum / period;
    }

    // ── EMA ───────────────────────────────────────────────────────────

    private static double ema(double[] arr, int period) {
        int n = arr.length;
        if (n == 0) return 0.0;
        if (n == 1) return arr[0];
        double k = 2.0 / (period + 1);
        double emaVal = arr[0];
        for (int i = 1; i < n; i++) {
            emaVal = arr[i] * k + emaVal * (1.0 - k);
        }
        return emaVal;
    }

    private static double[] rollingEmaSeries(double[] arr, int period) {
        int n = arr.length;
        if (n == 0) return new double[0];
        double[] out = new double[n];
        double k = 2.0 / (period + 1);
        out[0] = arr[0];
        for (int i = 1; i < n; i++) {
            out[i] = arr[i] * k + out[i - 1] * (1.0 - k);
        }
        return out;
    }

    // ── True Range ────────────────────────────────────────────────────

    private static double[] trueRange(double[] highs, double[] lows, double[] closes) {
        int n = highs.length;
        if (n < 2) return new double[0];
        double[] tr = new double[n - 1];
        for (int i = 1; i < n; i++) {
            double pc = closes[i - 1];
            tr[i - 1] = Math.max(highs[i] - lows[i],
                    Math.max(Math.abs(highs[i] - pc), Math.abs(lows[i] - pc)));
        }
        return tr;
    }

    // ── RSI ───────────────────────────────────────────────────────────

    private static double rsi(double[] closes, int period) {
        int n = closes.length;
        if (n < period + 1) return 50.0;
        double avgGain = 0.0;
        double avgLoss = 0.0;
        // Initial average
        for (int i = 0; i < period; i++) {
            double delta = closes[i + 1] - closes[i];
            if (delta > 0) avgGain += delta;
            else avgLoss += (-delta);
        }
        avgGain /= period;
        avgLoss /= period;
        // Wilder smoothing
        for (int i = period; i < n - 1; i++) {
            double delta = closes[i + 1] - closes[i];
            double gain = delta > 0 ? delta : 0.0;
            double loss = delta < 0 ? -delta : 0.0;
            avgGain = (avgGain * (period - 1) + gain) / period;
            avgLoss = (avgLoss * (period - 1) + loss) / period;
        }
        if (avgLoss == 0.0) return 100.0;
        double rs = avgGain / avgLoss;
        return 100.0 - 100.0 / (1.0 + rs);
    }

    // ── ATR ───────────────────────────────────────────────────────────

    private static double atr(double[] highs, double[] lows, double[] closes, int period) {
        double[] tr = trueRange(highs, lows, closes);
        if (tr.length == 0) return 0.0;
        if (tr.length < period) {
            double sum = 0.0;
            for (double v : tr) sum += v;
            return sum / tr.length;
        }
        // Initial ATR
        double atrVal = 0.0;
        for (int i = 0; i < period; i++) atrVal += tr[i];
        atrVal /= period;
        for (int i = period; i < tr.length; i++) {
            atrVal = (atrVal * (period - 1) + tr[i]) / period;
        }
        return atrVal;
    }

    // ── Stochastic ────────────────────────────────────────────────────

    private static double[] stochastic(double[] highs, double[] lows, double[] closes, int kPeriod, int dPeriod) {
        int n = closes.length;
        if (n < kPeriod) return new double[]{50.0, 50.0};

        List<Double> kValues = new ArrayList<>();
        for (int i = kPeriod - 1; i < n; i++) {
            double windowHigh = highs[i - kPeriod + 1];
            double windowLow = lows[i - kPeriod + 1];
            for (int j = i - kPeriod + 2; j <= i; j++) {
                windowHigh = Math.max(windowHigh, highs[j]);
                windowLow = Math.min(windowLow, lows[j]);
            }
            double denom = windowHigh - windowLow;
            double kVal = (denom == 0.0) ? 50.0 : (closes[i] - windowLow) / denom * 100.0;
            kValues.add(kVal);
        }

        double kNow = kValues.get(kValues.size() - 1);
        int dStart = Math.max(0, kValues.size() - dPeriod);
        double dSum = 0.0;
        int dCount = 0;
        for (int i = dStart; i < kValues.size(); i++) {
            dSum += kValues.get(i);
            dCount++;
        }
        double dNow = dCount > 0 ? dSum / dCount : 50.0;
        return new double[]{kNow, dNow};
    }

    // ── ADX ───────────────────────────────────────────────────────────

    private static double[] adx(double[] highs, double[] lows, double[] closes, int period) {
        int n = closes.length;
        if (n < period + 1) return new double[]{0.0, 0.0, 0.0};

        double[] tr = trueRange(highs, lows, closes);

        // Directional movement
        double[] upMove = new double[n - 1];
        double[] downMove = new double[n - 1];
        for (int i = 1; i < n; i++) {
            upMove[i - 1] = highs[i] - highs[i - 1];
            downMove[i - 1] = lows[i - 1] - lows[i];
        }

        double[] plusDm = new double[n - 1];
        double[] minusDm = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            plusDm[i] = (upMove[i] > downMove[i] && upMove[i] > 0) ? upMove[i] : 0.0;
            minusDm[i] = (downMove[i] > upMove[i] && downMove[i] > 0) ? downMove[i] : 0.0;
        }

        if (tr.length < period) return new double[]{0.0, 0.0, 0.0};

        double[] smoothTr = wilderSmooth(tr, period);
        double[] smoothPlusDm = wilderSmooth(plusDm, period);
        double[] smoothMinusDm = wilderSmooth(minusDm, period);

        int len = smoothTr.length;
        double[] plusDiRaw = new double[len];
        double[] minusDiRaw = new double[len];
        double[] dx = new double[len];

        for (int i = 0; i < len; i++) {
            plusDiRaw[i] = smoothTr[i] > 0 ? smoothPlusDm[i] / smoothTr[i] * 100.0 : 0.0;
            minusDiRaw[i] = smoothTr[i] > 0 ? smoothMinusDm[i] / smoothTr[i] * 100.0 : 0.0;
            double sum = plusDiRaw[i] + minusDiRaw[i];
            dx[i] = sum > 0 ? Math.abs(plusDiRaw[i] - minusDiRaw[i]) / sum * 100.0 : 0.0;
        }

        double adxVal;
        if (dx.length < period) {
            double sum = 0.0;
            for (double v : dx) sum += v;
            adxVal = dx.length > 0 ? sum / dx.length : 0.0;
        } else {
            adxVal = 0.0;
            for (int i = 0; i < period; i++) adxVal += dx[i];
            adxVal /= period;
            for (int i = period; i < dx.length; i++) {
                adxVal = (adxVal * (period - 1) + dx[i]) / period;
            }
        }

        double lastPlusDi = len > 0 ? plusDiRaw[len - 1] : 0.0;
        double lastMinusDi = len > 0 ? minusDiRaw[len - 1] : 0.0;
        return new double[]{adxVal, lastPlusDi, lastMinusDi};
    }

    private static double[] wilderSmooth(double[] arr, int period) {
        double[] result = new double[arr.length - period + 1];
        double s = 0.0;
        for (int i = 0; i < period; i++) s += arr[i];
        result[0] = s;
        for (int i = period; i < arr.length; i++) {
            s = s - s / period + arr[i];
            result[i - period + 1] = s;
        }
        return result;
    }

    // ── VWAP ──────────────────────────────────────────────────────────

    private static double vwap(double[] typical, double[] volumes) {
        if (volumes.length == 0) return typical.length > 0 ? typical[typical.length - 1] : 0.0;
        double cumTpVol = 0.0;
        double cumVol = 0.0;
        for (int i = 0; i < typical.length; i++) {
            cumTpVol += typical[i] * volumes[i];
            cumVol += volumes[i];
        }
        return cumVol > 0 ? cumTpVol / cumVol : (typical.length > 0 ? typical[typical.length - 1] : 0.0);
    }

    // ── Momentum ──────────────────────────────────────────────────────

    private static double momentum(double[] closes, int period) {
        int n = closes.length;
        if (n <= period || closes[n - 1 - period] == 0.0) return 0.0;
        return (closes[n - 1] / closes[n - 1 - period] - 1.0) * 100.0;
    }

    // ── Internal indicator computation ────────────────────────────────

    private static Map<String, Object> computeIndicators(BarBuffer buf, double price, double volume) {
        double[] opens = buf.getOpens();
        double[] highs = buf.getHighs();
        double[] lows = buf.getLows();
        double[] closes = buf.getCloses();
        double[] volumes = buf.getVolumes();
        double[] typical = buf.getTypicalPrices();
        int n = closes.length;

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("price", price);
        result.put("high", price);
        result.put("low", price);
        result.put("open", price);
        result.put("volume", volume);
        result.put("pair", "");

        if (n > 0) {
            result.put("open", opens[n - 1]);
            result.put("high", highs[n - 1]);
            result.put("low", lows[n - 1]);
        }

        // SMA
        result.put("sma_15", sma(closes, 15));
        result.put("sma_20", sma(closes, 20));

        // EMA
        double ema12 = ema(closes, 12);
        double ema26 = ema(closes, 26);
        result.put("ema_12", ema12);
        result.put("ema_26", ema26);

        // MACD (12, 26, 9)
        double macdLine = ema12 - ema26;
        double macdSignal = 0.0;
        if (n >= 2) {
            double[] ema12Series = rollingEmaSeries(closes, 12);
            double[] ema26Series = rollingEmaSeries(closes, 26);
            double[] macdSeries = new double[n];
            for (int i = 0; i < n; i++) {
                macdSeries[i] = ema12Series[i] - ema26Series[i];
            }
            double[] signalSeries = rollingEmaSeries(macdSeries, 9);
            macdSignal = signalSeries[n - 1];
        }
        double macdHist = macdLine - macdSignal;
        result.put("macd", macdLine);
        result.put("macd_signal", macdSignal);
        result.put("macd_hist", macdHist);

        // RSI(14)
        double rsiVal = rsi(closes, 14);
        result.put("rsi", rsiVal);
        result.put("rsi_14", rsiVal);

        // Bollinger Bands(20, 2)
        if (n >= 20) {
            double sum = 0.0;
            for (int i = n - 20; i < n; i++) sum += closes[i];
            double bbMid = sum / 20.0;
            double varSum = 0.0;
            for (int i = n - 20; i < n; i++) {
                double d = closes[i] - bbMid;
                varSum += d * d;
            }
            double bbStd = Math.sqrt(varSum / 20.0);
            result.put("bb_mid", bbMid);
            result.put("bb_upper", bbMid + 2.0 * bbStd);
            result.put("bb_lower", bbMid - 2.0 * bbStd);
        } else {
            double bbMid = n > 0 ? sma(closes, n) : price;
            result.put("bb_mid", bbMid);
            result.put("bb_upper", bbMid);
            result.put("bb_lower", bbMid);
        }

        // ATR(14)
        result.put("atr_14", atr(highs, lows, closes, 14));

        // Stochastic %K(14) / %D(3)
        double[] stoch = stochastic(highs, lows, closes, 14, 3);
        result.put("stoch_k", stoch[0]);
        result.put("stoch_d", stoch[1]);

        // ADX(14)
        double[] adxVals = adx(highs, lows, closes, 14);
        result.put("adx", adxVals[0]);
        result.put("adx_14", adxVals[0]);
        result.put("plus_di", adxVals[1]);
        result.put("minus_di", adxVals[2]);

        // VWAP
        result.put("vwap", vwap(typical, volumes));

        // Momentum (10-bar ROC)
        result.put("momentum", momentum(closes, 10));

        // Average volume (20-bar)
        result.put("avg_volume", sma(volumes, 20));

        // Log return
        if (n >= 2) {
            result.put("log_return", Math.log(closes[n - 1] / closes[n - 2]));
        } else {
            result.put("log_return", 0.0);
        }

        // Derive realistic high/low from ATR when bars are synthetic (tick-based).
        // Since BarBuffer.appendTick sets high==low==close for each tick,
        // range-based indicators (ATR, ADX) still compute correctly via
        // the |close[i] - close[i-1]| component of TrueRange.  However,
        // codecs that inspect (high - low) from the result map would see 0.
        // We patch the result map's high/low using ATR so codecs get useful values.
        double atrVal = (double) result.getOrDefault("atr_14", 0.0);
        double prevClose = n >= 2 ? closes[n - 2] : price;
        double move = price - prevClose;
        if (atrVal > 0) {
            double halfSpread = atrVal * 0.5;
            double syntheticHigh = Math.max(price, prevClose) + halfSpread * 0.3;
            double syntheticLow = Math.min(price, prevClose) - halfSpread * 0.3;
            result.put("high", syntheticHigh);
            result.put("low", syntheticLow);
        } else if (Math.abs(move) > 0) {
            result.put("high", Math.max(price, prevClose));
            result.put("low", Math.min(price, prevClose));
        }

        return result;
    }

    // ── Bar Buffer ────────────────────────────────────────────────────

    private static class BarBuffer {
        private final int maxlen;
        private final List<Double> opens = new ArrayList<>();
        private final List<Double> highs = new ArrayList<>();
        private final List<Double> lows = new ArrayList<>();
        private final List<Double> closes = new ArrayList<>();
        private final List<Double> volumes = new ArrayList<>();
        private final List<Double> typicalPrices = new ArrayList<>();

        BarBuffer(int maxlen) {
            this.maxlen = maxlen;
        }

        void appendTick(double price, double volume) {
            opens.add(price);
            highs.add(price);
            lows.add(price);
            closes.add(price);
            volumes.add(volume);
            typicalPrices.add(price);
            if (closes.size() > maxlen) {
                int excess = closes.size() - maxlen;
                for (int i = 0; i < excess; i++) {
                    opens.remove(0);
                    highs.remove(0);
                    lows.remove(0);
                    closes.remove(0);
                    volumes.remove(0);
                    typicalPrices.remove(0);
                }
            }
        }

        double[] getOpens() { return toArray(opens); }
        double[] getHighs() { return toArray(highs); }
        double[] getLows() { return toArray(lows); }
        double[] getCloses() { return toArray(closes); }
        double[] getVolumes() { return toArray(volumes); }
        double[] getTypicalPrices() { return toArray(typicalPrices); }

        private static double[] toArray(List<Double> list) {
            double[] arr = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = list.get(i);
            }
            return arr;
        }
    }
}
