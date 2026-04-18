package com.xtrade.kline;

import com.xtrade.PaperTradingEngine;
import com.xtrade.TradeRecord;
import com.xtrade.codec.BaseCodecExpert;
import com.xtrade.codec.CodecSignalSupport;
import com.xtrade.codec.IndicatorComputer;
import com.xtrade.codec.SignalResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Paper-trading agent that consumes canonical kline bars, computes showdown-style
 * indicators, runs a codec strategy, and executes resulting spot trades.
 */
public final class CodecPaperTradingAgent implements PaperTradingAgent {
    private final BaseCodecExpert codec;
    private final IndicatorComputer indicatorComputer;
    private final double convictionThreshold;
    private final double buyCashFraction;
    private final List<SignalObservation> observations = new ArrayList<>();
    private final List<TradeDecision> decisions = new ArrayList<>();
    private volatile long maxDecisionLatencyNanos;

    public CodecPaperTradingAgent(BaseCodecExpert codec) {
        this(codec, 0.40, 0.95, 512);
    }

    public CodecPaperTradingAgent(BaseCodecExpert codec,
                                  double convictionThreshold,
                                  double buyCashFraction,
                                  int indicatorBufferSize) {
        this.codec = Objects.requireNonNull(codec, "codec must not be null");
        if (convictionThreshold < 0.0 || convictionThreshold > 1.0) {
            throw new IllegalArgumentException("convictionThreshold must be in [0,1]");
        }
        if (buyCashFraction <= 0.0 || buyCashFraction > 1.0) {
            throw new IllegalArgumentException("buyCashFraction must be in (0,1]");
        }
        this.convictionThreshold = convictionThreshold;
        this.buyCashFraction = buyCashFraction;
        this.indicatorComputer = new IndicatorComputer(indicatorBufferSize);
    }

    @Override
    public void onConnected(KlineSeriesId seriesId, PaperTradingEngine engine) {
        codec.resetRuntimeState();
    }

    @Override
    public synchronized void onBar(KlineBar bar, PaperTradingEngine engine) {
        Map<String, Object> marketData = CodecSignalSupport.marketDataFromBar(indicatorComputer, bar);
        double price = bar.closePrice().doubleValue();
        codec.recordInstruments(CodecSignalSupport.extractInstrumentReadings(marketData, price));
        double[] indicatorVec = CodecSignalSupport.buildIndicatorVec(marketData);
        SignalResult signal = codec.forward(marketData, indicatorVec);
        observations.add(new SignalObservation(bar.openTimeMillis(), signal.getConviction(), signal.getDirection(), price));

        String decision = "HOLD";
        long started = System.nanoTime();
        if (signal.getConviction() >= convictionThreshold) {
            String asset = KlineSeriesIdAsset.extractBaseAsset(bar.seriesId().symbol());
            double held = engine.getHolding(asset);
            if (signal.getDirection() > 0.0 && held <= 1e-12) {
                double notional = engine.getCashBalance() * buyCashFraction * Math.max(signal.getConviction(), 0.1);
                double quantity = notional / price;
                if (quantity > 1e-12) {
                    TradeRecord record = engine.submitMarketOrder(bar.seriesId().symbol(), "BUY", quantity, bar);
                    decision = record.getSide().name();
                }
            } else if (signal.getDirection() < 0.0 && held > 1e-12) {
                TradeRecord record = engine.submitMarketOrder(bar.seriesId().symbol(), "SELL", held, bar);
                decision = record.getSide().name();
            }
        }
        long elapsed = System.nanoTime() - started;
        maxDecisionLatencyNanos = Math.max(maxDecisionLatencyNanos, elapsed);
        decisions.add(new TradeDecision(bar.openTimeMillis(), decision, signal.getConviction(), signal.getDirection(), elapsed));
    }

    public BaseCodecExpert getCodec() {
        return codec;
    }

    public synchronized List<SignalObservation> getObservations() {
        return Collections.unmodifiableList(new ArrayList<>(observations));
    }

    public synchronized List<TradeDecision> getDecisions() {
        return Collections.unmodifiableList(new ArrayList<>(decisions));
    }

    public long getMaxDecisionLatencyNanos() {
        return maxDecisionLatencyNanos;
    }

    public synchronized long decisionCount(String action) {
        return decisions.stream().filter(d -> d.action.equals(action)).count();
    }

    public static final class SignalObservation {
        private final long barOpenTimeMillis;
        private final double conviction;
        private final double direction;
        private final double closePrice;

        private SignalObservation(long barOpenTimeMillis, double conviction, double direction, double closePrice) {
            this.barOpenTimeMillis = barOpenTimeMillis;
            this.conviction = conviction;
            this.direction = direction;
            this.closePrice = closePrice;
        }

        public long getBarOpenTimeMillis() {
            return barOpenTimeMillis;
        }

        public double getConviction() {
            return conviction;
        }

        public double getDirection() {
            return direction;
        }

        public double getClosePrice() {
            return closePrice;
        }
    }

    public static final class TradeDecision {
        private final long barOpenTimeMillis;
        private final String action;
        private final double conviction;
        private final double direction;
        private final long latencyNanos;

        private TradeDecision(long barOpenTimeMillis, String action, double conviction, double direction, long latencyNanos) {
            this.barOpenTimeMillis = barOpenTimeMillis;
            this.action = action;
            this.conviction = conviction;
            this.direction = direction;
            this.latencyNanos = latencyNanos;
        }

        public long getBarOpenTimeMillis() {
            return barOpenTimeMillis;
        }

        public String getAction() {
            return action;
        }

        public double getConviction() {
            return conviction;
        }

        public double getDirection() {
            return direction;
        }

        public long getLatencyNanos() {
            return latencyNanos;
        }
    }

    private static final class KlineSeriesIdAsset {
        private static String extractBaseAsset(String pair) {
            int idx = pair.indexOf('/');
            return idx > 0 ? pair.substring(0, idx) : pair;
        }
    }
}
