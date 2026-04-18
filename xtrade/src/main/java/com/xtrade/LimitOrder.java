package com.xtrade;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

/**
 * Represents a limit order stored in the paper trading order book.
 * Limit orders rest until the market price meets or crosses the limit price.
 */
public class LimitOrder {

    public enum Status {
        OPEN, FILLED, CANCELLED
    }

    private final String orderId;
    private final Instant createdAt;
    private final String pair;
    private final TradeRecord.Side side;
    private final BigDecimal limitPrice;
    private final BigDecimal quantity;
    private volatile Status status;

    public LimitOrder(String pair, TradeRecord.Side side, BigDecimal limitPrice, BigDecimal quantity) {
        this(UUID.randomUUID().toString(), Instant.now(), pair, side, limitPrice, quantity, Status.OPEN);
    }

    public LimitOrder(String orderId,
                      Instant createdAt,
                      String pair,
                      TradeRecord.Side side,
                      BigDecimal limitPrice,
                      BigDecimal quantity,
                      Status status) {
        this.orderId = orderId;
        this.createdAt = createdAt;
        this.pair = pair;
        this.side = side;
        this.limitPrice = limitPrice;
        this.quantity = quantity;
        this.status = status;
    }

    public String getOrderId() { return orderId; }
    public Instant getCreatedAt() { return createdAt; }
    public String getPair() { return pair; }
    public TradeRecord.Side getSide() { return side; }
    public BigDecimal getLimitPrice() { return limitPrice; }
    public BigDecimal getQuantity() { return quantity; }
    public Status getStatus() { return status; }

    void markFilled() { this.status = Status.FILLED; }
    void markCancelled() { this.status = Status.CANCELLED; }

    @Override
    public String toString() {
        return String.format("LimitOrder{id=%s, pair=%s, side=%s, limitPrice=%s, qty=%s, status=%s}",
                orderId, pair, side, limitPrice, quantity, status);
    }
}
