package com.xtrade;

/**
 * Snapshot of a single asset's position within the portfolio.
 */
public class PortfolioPosition {

    private final String pair;
    private final String baseAsset;
    private final double quantity;
    private final double averageCost;
    private final double currentPrice;
    private final double unrealizedPnl;
    private final double unrealizedPnlPercent;
    private final double totalFeesPaid;
    private final double totalRealizedPnl;
    private final double usdValue;

    public PortfolioPosition(String pair, String baseAsset, double quantity, double averageCost,
                             double currentPrice, double unrealizedPnl, double unrealizedPnlPercent,
                             double totalFeesPaid, double totalRealizedPnl, double usdValue) {
        this.pair = pair;
        this.baseAsset = baseAsset;
        this.quantity = quantity;
        this.averageCost = averageCost;
        this.currentPrice = currentPrice;
        this.unrealizedPnl = unrealizedPnl;
        this.unrealizedPnlPercent = unrealizedPnlPercent;
        this.totalFeesPaid = totalFeesPaid;
        this.totalRealizedPnl = totalRealizedPnl;
        this.usdValue = usdValue;
    }

    public String getPair() { return pair; }
    public String getBaseAsset() { return baseAsset; }
    public double getQuantity() { return quantity; }
    public double getAverageCost() { return averageCost; }
    public double getCurrentPrice() { return currentPrice; }
    public double getUnrealizedPnl() { return unrealizedPnl; }
    public double getUnrealizedPnlPercent() { return unrealizedPnlPercent; }
    public double getTotalFeesPaid() { return totalFeesPaid; }
    public double getTotalRealizedPnl() { return totalRealizedPnl; }
    public double getUsdValue() { return usdValue; }

    @Override
    public String toString() {
        return String.format(
                "PortfolioPosition{asset=%s, qty=%.8f, avgCost=%.4f, curPrice=%.4f, " +
                "unrealizedPnl=%.4f (%.2f%%), realizedPnl=%.4f, fees=%.4f, usdValue=%.4f}",
                baseAsset, quantity, averageCost, currentPrice, unrealizedPnl,
                unrealizedPnlPercent, totalRealizedPnl, totalFeesPaid, usdValue);
    }
}
