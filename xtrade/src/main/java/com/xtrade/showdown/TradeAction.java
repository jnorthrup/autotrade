package com.xtrade.showdown;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Represents a trade action produced by a ShowdownAgent on a single tick.
 * Mirrors the Python make_trade_action() dict.
 *
 * Includes the indicator values (RSI, MACD histogram, BB position, ADX,
 * VWAP, momentum, ATR) that were computed at decision time.
 */
public class TradeAction {

    public static final String BUY = "BUY";
    public static final String SELL = "SELL";
    public static final String HOLD = "HOLD";

    private final String pair;
    private final String action;    // BUY, SELL, or HOLD
    private final double size;      // quantity traded
    private final double price;     // execution price
    private final double conviction;
    private final double direction;
    private final Map<String, Object> indicators; // indicator snapshot at decision time

    public TradeAction(String pair, String action, double size, double price,
                       double conviction, double direction,
                       Map<String, Object> indicators) {
        this.pair = pair;
        this.action = action;
        this.size = size;
        this.price = price;
        this.conviction = conviction;
        this.direction = direction;
        this.indicators = indicators != null
                ? Collections.unmodifiableMap(new LinkedHashMap<>(indicators))
                : Collections.emptyMap();
    }

    /** Backward-compatible constructor (no indicator snapshot). */
    public TradeAction(String pair, String action, double size, double price,
                       double conviction, double direction) {
        this(pair, action, size, price, conviction, direction, null);
    }

    public String getPair() { return pair; }
    public String getAction() { return action; }
    public double getSize() { return size; }
    public double getPrice() { return price; }
    public double getConviction() { return conviction; }
    public double getDirection() { return direction; }

    /**
     * Returns the indicator readings used at the decision time for this trade.
     * Keys include: rsi, macd_hist, bb_position, adx, vwap, momentum, atr_14,
     * macd, macd_signal, bb_upper, bb_lower, bb_mid, plus_di, minus_di,
     * stoch_k, stoch_d, sma_20, ema_12, log_return, etc.
     */
    public Map<String, Object> getIndicators() { return indicators; }

    @Override
    public String toString() {
        return String.format("TradeAction{pair=%s, action=%s, size=%.6f, price=%.2f, conviction=%.4f, direction=%.4f, indicators=%d}",
                pair, action, size, price, conviction, direction, indicators.size());
    }
}
