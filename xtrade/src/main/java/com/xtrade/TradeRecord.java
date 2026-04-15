package com.xtrade;

import java.math.BigDecimal;
import java.time.Instant;

/**
 * Immutable record of a single executed trade in the paper trading engine.
 */
public class TradeRecord {

    private final Instant timestamp;
    private final String pair;
    private final Side side;
    private final BigDecimal price;
    private final BigDecimal quantity;
    private final BigDecimal fee;
    private final BigDecimal totalCost; // price * quantity + fee for BUY, price * quantity - fee for SELL

    public enum Side {
        BUY, SELL
    }

    public TradeRecord(Instant timestamp, String pair, Side side, BigDecimal price, BigDecimal quantity, BigDecimal fee, BigDecimal totalCost) {
        this.timestamp = timestamp;
        this.pair = pair;
        this.side = side;
        this.price = price;
        this.quantity = quantity;
        this.fee = fee;
        this.totalCost = totalCost;
    }

    public Instant getTimestamp() { return timestamp; }
    public String getPair() { return pair; }
    public Side getSide() { return side; }
    public BigDecimal getPrice() { return price; }
    public BigDecimal getQuantity() { return quantity; }
    public BigDecimal getFee() { return fee; }
    public BigDecimal getTotalCost() { return totalCost; }

    @Override
    public String toString() {
        return String.format("TradeRecord{ts=%s, pair=%s, side=%s, price=%s, qty=%s, fee=%s, cost=%s}",
                timestamp, pair, side, price, quantity, fee, totalCost);
    }
}
