package com.xtrade;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Serializable state container for the paper trading engine.
 * Used to persist and restore portfolio state to/from JSON.
 */
public class PortfolioState {

    private double startingBalance;
    private double cashBalance;
    private Map<String, Double> holdings;
    private Map<String, Double> averageCost;
    private Map<String, Double> realizedPnl;
    private Map<String, Double> feesPaid;
    private Map<String, Double> marketPrices;
    private List<TradeRecord> tradeHistory;
    private List<LimitOrderState> openOrders;

    /** No-args constructor for Gson deserialization. */
    public PortfolioState() {
        this.holdings = new HashMap<>();
        this.averageCost = new HashMap<>();
        this.realizedPnl = new HashMap<>();
        this.feesPaid = new HashMap<>();
        this.marketPrices = new HashMap<>();
        this.tradeHistory = new ArrayList<>();
        this.openOrders = new ArrayList<>();
    }

    public PortfolioState(double startingBalance, double cashBalance,
                          Map<String, Double> holdings, Map<String, Double> averageCost,
                          Map<String, Double> realizedPnl, Map<String, Double> feesPaid,
                          Map<String, Double> marketPrices,
                          List<TradeRecord> tradeHistory, List<LimitOrderState> openOrders) {
        this.startingBalance = startingBalance;
        this.cashBalance = cashBalance;
        this.holdings = holdings;
        this.averageCost = averageCost;
        this.realizedPnl = realizedPnl;
        this.feesPaid = feesPaid;
        this.marketPrices = marketPrices;
        this.tradeHistory = tradeHistory;
        this.openOrders = openOrders;
    }

    public double getStartingBalance() { return startingBalance; }
    public void setStartingBalance(double startingBalance) { this.startingBalance = startingBalance; }

    public double getCashBalance() { return cashBalance; }
    public void setCashBalance(double cashBalance) { this.cashBalance = cashBalance; }

    public Map<String, Double> getHoldings() { return holdings; }
    public void setHoldings(Map<String, Double> holdings) { this.holdings = holdings; }

    public Map<String, Double> getAverageCost() { return averageCost; }
    public void setAverageCost(Map<String, Double> averageCost) { this.averageCost = averageCost; }

    public Map<String, Double> getRealizedPnl() { return realizedPnl; }
    public void setRealizedPnl(Map<String, Double> realizedPnl) { this.realizedPnl = realizedPnl; }

    public Map<String, Double> getFeesPaid() { return feesPaid; }
    public void setFeesPaid(Map<String, Double> feesPaid) { this.feesPaid = feesPaid; }

    public Map<String, Double> getMarketPrices() { return marketPrices; }
    public void setMarketPrices(Map<String, Double> marketPrices) { this.marketPrices = marketPrices; }

    public List<TradeRecord> getTradeHistory() { return tradeHistory; }
    public void setTradeHistory(List<TradeRecord> tradeHistory) { this.tradeHistory = tradeHistory; }

    public List<LimitOrderState> getOpenOrders() { return openOrders; }
    public void setOpenOrders(List<LimitOrderState> openOrders) { this.openOrders = openOrders; }

    /**
     * Serializable representation of a limit order (without the volatile status field
     * and with a simpler structure suitable for JSON).
     */
    public static class LimitOrderState {
        private String orderId;
        private String createdAt;
        private String pair;
        private String side;
        private double limitPrice;
        private double quantity;
        private String status;

        public LimitOrderState() {}

        public LimitOrderState(String orderId, String createdAt, String pair,
                               String side, double limitPrice, double quantity, String status) {
            this.orderId = orderId;
            this.createdAt = createdAt;
            this.pair = pair;
            this.side = side;
            this.limitPrice = limitPrice;
            this.quantity = quantity;
            this.status = status;
        }

        public String getOrderId() { return orderId; }
        public String getCreatedAt() { return createdAt; }
        public String getPair() { return pair; }
        public String getSide() { return side; }
        public double getLimitPrice() { return limitPrice; }
        public double getQuantity() { return quantity; }
        public String getStatus() { return status; }
    }
}
