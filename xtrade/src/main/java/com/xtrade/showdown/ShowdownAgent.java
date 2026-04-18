package com.xtrade.showdown;

import com.xtrade.codec.BaseCodecExpert;
import com.xtrade.codec.IndicatorComputer;
import com.xtrade.codec.SignalResult;

import java.util.*;

/**
 * Agent adapter that wraps a BaseCodecExpert codec for the showdown harness.
 *
 * Each agent maintains fully isolated state:
 *   - Its own IndicatorComputer (rolling indicator buffers per pair)
 *   - A virtual portfolio: cash + holdings (pair -> qty)
 *   - Trade history (each entry includes indicator readings at decision time)
 *   - Per-tick indicator snapshots for the instrument context map
 *
 * Translates (conviction, direction) from codec.forward() into BUY/SELL/HOLD
 * with position sizing, mirroring the Python Agent.on_tick() logic.
 */
public class ShowdownAgent {

    private final BaseCodecExpert codec;
    private final double initialCash;
    private final double convictionThreshold;
    private final double positionFraction;

    // Isolated indicator computer
    private final IndicatorComputer indicatorComputer;

    // Virtual portfolio
    private double cash;
    private final Map<String, Double> holdings; // pair -> quantity held

    // Trade history (with indicator values)
    private final List<TradeAction> tradeHistory;

    // Per-tick indicator snapshots: tickIndex -> { pair -> { indicator -> value } }
    private final List<Map<String, Map<String, Object>>> indicatorSnapshots;

    /**
     * @param codec                the codec expert strategy
     * @param initialCash          starting virtual cash
     * @param convictionThreshold  minimum conviction to act (default 0.4)
     * @param positionFraction     fraction of cash to use per BUY (default 0.25)
     */
    public ShowdownAgent(BaseCodecExpert codec, double initialCash,
                         double convictionThreshold, double positionFraction) {
        this.codec = codec;
        this.initialCash = initialCash;
        this.convictionThreshold = convictionThreshold;
        this.positionFraction = positionFraction;

        this.indicatorComputer = new IndicatorComputer(200);
        this.cash = initialCash;
        this.holdings = new LinkedHashMap<>();
        this.tradeHistory = new ArrayList<>();
        this.indicatorSnapshots = new ArrayList<>();
    }

    public ShowdownAgent(BaseCodecExpert codec, double initialCash) {
        this(codec, initialCash, 0.4, 0.25);
    }

    /**
     * Process one tick of data and return a list of trade actions.
     * Mirrors Python Agent.on_tick().
     *
     * Records computed indicator readings into the instrument context map
     * (mirroring Python's record_instruments()) and stores per-tick indicator
     * snapshots for downstream access.
     *
     * @param tickData the tick data containing price/volume for each pair
     * @return list of TradeAction (one per pair)
     */
    public List<TradeAction> onTick(TickData tickData) {
        List<TradeAction> actions = new ArrayList<>();
        Map<String, Map<String, Object>> tickIndicatorSnapshot = new LinkedHashMap<>();

        for (Map.Entry<String, TickData.PairTick> entry : tickData.getTicks().entrySet()) {
            String pair = entry.getKey();
            double price = entry.getValue().getPrice();
            double volume = entry.getValue().getVolume();

            // 1. Compute indicators via isolated IndicatorComputer
            Map<String, Object> marketData = indicatorComputer.compute(pair, price, volume);

            // 2. Extract key indicator values into the instrument context map
            //    (mirrors Python's record_instruments())
            Map<String, Double> instrumentReadings = extractInstrumentReadings(marketData, price);
            codec.recordInstruments(instrumentReadings);

            // 3. Store per-pair indicator snapshot for this tick
            Map<String, Object> pairSnapshot = new LinkedHashMap<>(marketData);
            pairSnapshot.put("bb_position", computeBbPosition(marketData, price));
            pairSnapshot.put("vwap_ratio", computeVwapRatio(marketData, price));
            tickIndicatorSnapshot.put(pair, pairSnapshot);

            // 4. Build 64-element indicator vector
            double[] indicatorVec = buildIndicatorVec(marketData);

            // 5. Feed to codec forward
            SignalResult result = codec.forward(marketData, indicatorVec);
            double conviction = result.getConviction();
            double direction = result.getDirection();

            // 6. Translate (conviction, direction) -> BUY / SELL / HOLD
            String actionStr = TradeAction.HOLD;
            double size = 0.0;

            if (conviction > convictionThreshold) {
                if (direction > 0) {
                    // BUY signal
                    actionStr = TradeAction.BUY;
                    double spend = cash * positionFraction * conviction;
                    if (spend > 0 && price > 0) {
                        size = spend / price;
                        cash -= size * price;
                        holdings.merge(pair, size, Double::sum);
                    }
                } else if (direction < 0) {
                    // SELL signal
                    actionStr = TradeAction.SELL;
                    double held = holdings.getOrDefault(pair, 0.0);
                    if (held > 0 && price > 0) {
                        double sellFrac = conviction * positionFraction;
                        size = held * sellFrac;
                        if (size > 0) {
                            cash += size * price;
                            double remaining = held - size;
                            if (remaining < 1e-12) {
                                remaining = 0.0;
                            }
                            holdings.put(pair, remaining);
                        }
                    }
                }
            }

            // 7. Build indicator map for the trade action
            Map<String, Object> decisionIndicators = buildDecisionIndicatorMap(marketData, price);

            TradeAction action = new TradeAction(pair, actionStr, size, price,
                    conviction, direction, decisionIndicators);
            actions.add(action);
            tradeHistory.add(action);
        }

        // Store this tick's indicator snapshot
        indicatorSnapshots.add(tickIndicatorSnapshot);

        return actions;
    }

    /**
     * Extract key indicator readings from market data into a flat map
     * suitable for record_instruments(). Mirrors Python's record_instruments().
     */
    private Map<String, Double> extractInstrumentReadings(Map<String, Object> md, double price) {
        Map<String, Double> readings = new LinkedHashMap<>();
        readings.put("rsi", getDouble(md, "rsi", 50.0));
        readings.put("macd_hist", getDouble(md, "macd_hist", 0.0));
        readings.put("atr_14", getDouble(md, "atr_14", 0.0));
        readings.put("bb_position", computeBbPosition(md, price));
        readings.put("adx", getDouble(md, "adx", 0.0));
        readings.put("vwap_ratio", computeVwapRatio(md, price));
        readings.put("momentum", getDouble(md, "momentum", 0.0));
        readings.put("macd", getDouble(md, "macd", 0.0));
        readings.put("macd_signal", getDouble(md, "macd_signal", 0.0));
        readings.put("plus_di", getDouble(md, "plus_di", 0.0));
        readings.put("minus_di", getDouble(md, "minus_di", 0.0));
        readings.put("stoch_k", getDouble(md, "stoch_k", 50.0));
        readings.put("stoch_d", getDouble(md, "stoch_d", 50.0));
        return readings;
    }

    /**
     * Compute Bollinger Band position: (price - lower) / (upper - lower).
     * Returns 0.5 when BB width is zero.
     */
    static double computeBbPosition(Map<String, Object> md, double price) {
        double bbUpper = getDouble(md, "bb_upper", price);
        double bbLower = getDouble(md, "bb_lower", price);
        double bbWidth = bbUpper - bbLower;
        if (bbWidth > 1e-12) {
            return (price - bbLower) / bbWidth;
        }
        return 0.5;
    }

    /**
     * Compute VWAP ratio: price / VWAP. Returns 1.0 when VWAP is zero.
     */
    static double computeVwapRatio(Map<String, Object> md, double price) {
        double vwap = getDouble(md, "vwap", price);
        return vwap != 0.0 ? price / vwap : 1.0;
    }

    /**
     * Build the indicator map attached to each TradeAction, containing
     * the key indicator values at decision time.
     */
    private Map<String, Object> buildDecisionIndicatorMap(Map<String, Object> md, double price) {
        Map<String, Object> indicators = new LinkedHashMap<>();
        indicators.put("rsi", getDouble(md, "rsi", 50.0));
        indicators.put("macd_hist", getDouble(md, "macd_hist", 0.0));
        indicators.put("bb_position", computeBbPosition(md, price));
        indicators.put("adx", getDouble(md, "adx", 0.0));
        indicators.put("vwap", getDouble(md, "vwap", price));
        indicators.put("momentum", getDouble(md, "momentum", 0.0));
        indicators.put("atr_14", getDouble(md, "atr_14", 0.0));
        indicators.put("macd", getDouble(md, "macd", 0.0));
        indicators.put("macd_signal", getDouble(md, "macd_signal", 0.0));
        indicators.put("bb_upper", getDouble(md, "bb_upper", price));
        indicators.put("bb_lower", getDouble(md, "bb_lower", price));
        indicators.put("bb_mid", getDouble(md, "bb_mid", price));
        indicators.put("plus_di", getDouble(md, "plus_di", 0.0));
        indicators.put("minus_di", getDouble(md, "minus_di", 0.0));
        indicators.put("stoch_k", getDouble(md, "stoch_k", 50.0));
        indicators.put("stoch_d", getDouble(md, "stoch_d", 50.0));
        indicators.put("sma_20", getDouble(md, "sma_20", price));
        indicators.put("ema_12", getDouble(md, "ema_12", price));
        indicators.put("log_return", getDouble(md, "log_return", 0.0));
        return indicators;
    }

    /**
     * Build the 64-element indicator feature vector from market data.
     * Mirrors Python build_indicator_vec().
     */
    static double[] buildIndicatorVec(Map<String, Object> md) {
        double[] vec = new double[64];

        double price = getDouble(md, "price", 1.0);
        double safePrice = price != 0.0 ? price : 1.0;

        vec[0] = price / getDoubleNonZero(md, "sma_20", price);
        vec[1] = price / getDoubleNonZero(md, "sma_15", price);
        vec[2] = price / getDoubleNonZero(md, "ema_12", price);
        vec[3] = price / getDoubleNonZero(md, "ema_26", price);

        vec[4] = getDouble(md, "macd", 0.0) / safePrice;
        vec[5] = getDouble(md, "macd_signal", 0.0) / safePrice;
        vec[6] = getDouble(md, "macd_hist", 0.0) / safePrice;

        vec[7] = getDouble(md, "rsi", 50.0) / 100.0;

        double bbUpper = getDouble(md, "bb_upper", price);
        double bbLower = getDouble(md, "bb_lower", price);
        double bbMid = getDouble(md, "bb_mid", price);
        double bbWidth = bbUpper - bbLower;
        if (bbWidth > 0) {
            vec[8] = (price - bbLower) / bbWidth;
        } else {
            vec[8] = 0.5;
        }
        vec[9] = bbMid != 0.0 ? bbWidth / bbMid : 0.0;

        vec[10] = getDouble(md, "atr_14", 0.0) / safePrice;
        vec[11] = getDouble(md, "stoch_k", 50.0) / 100.0;
        vec[12] = getDouble(md, "stoch_d", 50.0) / 100.0;
        vec[13] = getDouble(md, "adx", 0.0) / 100.0;
        vec[14] = getDouble(md, "plus_di", 0.0) / 100.0;
        vec[15] = getDouble(md, "minus_di", 0.0) / 100.0;

        double vwap = getDouble(md, "vwap", price);
        vec[16] = vwap != 0.0 ? price / vwap : 1.0;

        vec[17] = getDouble(md, "momentum", 0.0) / 100.0;

        double avgVol = getDouble(md, "avg_volume", 0.0);
        double vol = getDouble(md, "volume", 0.0);
        vec[18] = avgVol > 0.0 ? vol / avgVol : 1.0;

        vec[19] = getDouble(md, "log_return", 0.0);

        // Slots 20-63 remain zero
        return vec;
    }

    private static double getDouble(Map<String, Object> md, String key, double def) {
        Object val = md.get(key);
        if (val == null) return def;
        if (val instanceof Number) return ((Number) val).doubleValue();
        return def;
    }

    private static double getDoubleNonZero(Map<String, Object> md, String key, double fallback) {
        double val = getDouble(md, key, fallback);
        return val != 0.0 ? val : fallback;
    }

    // ── Accessors ─────────────────────────────────────────────────────

    public double getCash() { return cash; }
    public Map<String, Double> getHoldings() { return Collections.unmodifiableMap(holdings); }
    public List<TradeAction> getTradeHistory() { return Collections.unmodifiableList(tradeHistory); }
    public BaseCodecExpert getCodec() { return codec; }
    public String getCodecName() { return codec.getName(); }
    public double getInitialCash() { return initialCash; }

    /**
     * Per-tick indicator snapshots: each entry maps pair -> indicator -> value.
     * Index in list corresponds to tick index.
     */
    public List<Map<String, Map<String, Object>>> getIndicatorSnapshots() {
        return Collections.unmodifiableList(indicatorSnapshots);
    }

    /**
     * Compute total portfolio value at current prices.
     */
    public double portfolioValue(Map<String, Double> prices) {
        double total = cash;
        for (Map.Entry<String, Double> entry : holdings.entrySet()) {
            total += entry.getValue() * prices.getOrDefault(entry.getKey(), 0.0);
        }
        return total;
    }

    /**
     * Reset agent to initial state.
     */
    public void reset() {
        cash = initialCash;
        holdings.clear();
        tradeHistory.clear();
        indicatorSnapshots.clear();
        codec.resetRuntimeState();
        codec.resetTradeLedger();
    }

    @Override
    public String toString() {
        return String.format("Agent(codec=%s, cash=%.2f, positions=%d)",
                codec.getName(), cash, holdings.size());
    }
}
