package com.xtrade;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Immutable snapshot of the entire paper trading portfolio.
 */
public class PortfolioSnapshot {

    private final double cashBalance;
    private final double totalPortfolioValueUsd;
    private final double totalUnrealizedPnl;
    private final double totalRealizedPnl;
    private final double totalFeesPaid;
    private final Map<String, PortfolioPosition> positions;
    private final List<TradeRecord> tradeHistory;

    public PortfolioSnapshot(double cashBalance, double totalPortfolioValueUsd,
                             double totalUnrealizedPnl, double totalRealizedPnl,
                             double totalFeesPaid,
                             Map<String, PortfolioPosition> positions,
                             List<TradeRecord> tradeHistory) {
        this.cashBalance = cashBalance;
        this.totalPortfolioValueUsd = totalPortfolioValueUsd;
        this.totalUnrealizedPnl = totalUnrealizedPnl;
        this.totalRealizedPnl = totalRealizedPnl;
        this.totalFeesPaid = totalFeesPaid;
        this.positions = Collections.unmodifiableMap(positions);
        this.tradeHistory = Collections.unmodifiableList(tradeHistory);
    }

    public double getCashBalance() { return cashBalance; }
    public double getTotalPortfolioValueUsd() { return totalPortfolioValueUsd; }
    public double getTotalUnrealizedPnl() { return totalUnrealizedPnl; }
    public double getTotalRealizedPnl() { return totalRealizedPnl; }
    public double getTotalFeesPaid() { return totalFeesPaid; }
    public Map<String, PortfolioPosition> getPositions() { return positions; }
    public List<TradeRecord> getTradeHistory() { return tradeHistory; }

    @Override
    public String toString() {
        return String.format(
                "PortfolioSnapshot{cash=%.2f, totalUsd=%.2f, unrealizedPnl=%.4f, " +
                "realizedPnl=%.4f, fees=%.4f, positions=%d, trades=%d}",
                cashBalance, totalPortfolioValueUsd, totalUnrealizedPnl,
                totalRealizedPnl, totalFeesPaid, positions.size(), tradeHistory.size());
    }
}
