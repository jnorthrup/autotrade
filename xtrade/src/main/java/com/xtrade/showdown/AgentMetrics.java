package com.xtrade.showdown;

import java.util.*;

/**
 * Per-agent performance metrics accumulator.
 *
 * Tracks equity curve, realized P&L via FIFO cost-basis, unrealized P&L,
 * Sharpe estimate, max drawdown, hit rate, and trade count.
 * Mirrors the Python AgentMetrics class.
 */
public class AgentMetrics {

    private final String agentName;
    private final double initialCash;

    // FIFO cost basis: pair -> list of [qty, price]
    private final Map<String, List<double[]>> costLots;

    private double realizedPnl;
    private final List<Double> equityCurve;
    private final List<Map<String, Object>> snapshots;
    private int totalTrades;
    private int winningTrades;
    private int roundTripTrades;

    public AgentMetrics(String agentName, double initialCash) {
        this.agentName = agentName;
        this.initialCash = initialCash;
        this.costLots = new LinkedHashMap<>();
        this.equityCurve = new ArrayList<>();
        this.equityCurve.add(initialCash);
        this.snapshots = new ArrayList<>();
        this.realizedPnl = 0.0;
        this.totalTrades = 0;
        this.winningTrades = 0;
        this.roundTripTrades = 0;
    }

    /**
     * Process one tick's actions and record a snapshot.
     *
     * @param cash      current cash balance
     * @param holdings  current holdings map (pair -> qty)
     * @param prices    current prices (pair -> price)
     * @param actions   list of TradeAction for this tick
     * @param tick      tick index
     * @param timestamp timestamp value
     * @return snapshot map for this tick
     */
    public Map<String, Object> recordTick(
            double cash,
            Map<String, Double> holdings,
            Map<String, Double> prices,
            List<TradeAction> actions,
            int tick,
            double timestamp
    ) {
        // Process each action for FIFO cost-basis tracking
        for (TradeAction act : actions) {
            String pair = act.getPair();
            String action = act.getAction();
            double size = act.getSize();
            double price = act.getPrice();

            if (TradeAction.BUY.equals(action) && size > 0) {
                costLots.computeIfAbsent(pair, k -> new ArrayList<>())
                        .add(new double[]{size, price});
                totalTrades++;
            } else if (TradeAction.SELL.equals(action) && size > 0) {
                double remaining = size;
                double tradePnl = 0.0;
                List<double[]> lots = costLots.getOrDefault(pair, new ArrayList<>());

                while (remaining > 1e-15 && !lots.isEmpty()) {
                    double[] lot = lots.get(0);
                    double lotQty = lot[0];
                    double lotPrice = lot[1];
                    double filled = Math.min(remaining, lotQty);
                    tradePnl += (price - lotPrice) * filled;
                    remaining -= filled;
                    lotQty -= filled;
                    if (lotQty < 1e-15) {
                        lots.remove(0);
                    } else {
                        lot[0] = lotQty;
                    }
                }

                realizedPnl += tradePnl;
                totalTrades++;
                roundTripTrades++;
                if (tradePnl > 0) {
                    winningTrades++;
                }
            }
        }

        // Mark-to-market: compute total portfolio value
        double holdingsValue = 0.0;
        for (Map.Entry<String, Double> entry : holdings.entrySet()) {
            holdingsValue += entry.getValue() * prices.getOrDefault(entry.getKey(), 0.0);
        }
        double totalValue = cash + holdingsValue;
        double unrealizedPnl = totalValue - initialCash;
        equityCurve.add(totalValue);

        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("tick", tick);
        snapshot.put("timestamp", timestamp);
        snapshot.put("cash", cash);
        snapshot.put("holdings_value", holdingsValue);
        snapshot.put("total_value", totalValue);
        snapshot.put("unrealized_pnl", unrealizedPnl);
        snapshot.put("realized_pnl", realizedPnl);
        snapshot.put("trade_count", totalTrades);
        snapshots.add(snapshot);

        return snapshot;
    }

    /**
     * Compute aggregate metrics over the full recorded history.
     * Mirrors Python AgentMetrics.compute_summary().
     *
     * @return map with keys: agent_name, initial_cash, final_value,
     *         total_pnl, return_pct, realized_pnl, unrealized_pnl,
     *         sharpe_estimate, hit_rate, trade_count, max_drawdown,
     *         max_drawdown_pct, ticks_processed
     */
    public Map<String, Object> computeSummary() {
        double finalValue = equityCurve.isEmpty() ? initialCash : equityCurve.get(equityCurve.size() - 1);
        double totalPnl = finalValue - initialCash;
        double returnPct = initialCash != 0.0 ? (totalPnl / initialCash) * 100.0 : 0.0;

        // Sharpe estimate (annualized assuming ~86400 ticks/day * 252 trading days)
        double sharpe = computeSharpe();

        // Max drawdown
        double[] ddResult = computeMaxDrawdown();
        double maxDrawdown = ddResult[0];
        double maxDrawdownPct = ddResult[1];

        double hitRate = roundTripTrades > 0 ? (double) winningTrades / roundTripTrades : 0.0;

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("agent_name", agentName);
        summary.put("initial_cash", initialCash);
        summary.put("final_value", finalValue);
        summary.put("total_pnl", totalPnl);
        summary.put("return_pct", returnPct);
        summary.put("realized_pnl", realizedPnl);
        summary.put("unrealized_pnl", finalValue - initialCash - realizedPnl);
        summary.put("sharpe_estimate", sharpe);
        summary.put("hit_rate", hitRate);
        summary.put("trade_count", totalTrades);
        summary.put("max_drawdown", maxDrawdown);
        summary.put("max_drawdown_pct", maxDrawdownPct);
        summary.put("ticks_processed", snapshots.size());
        return summary;
    }

    private double computeSharpe() {
        if (equityCurve.size() <= 1) return 0.0;

        int n = equityCurve.size() - 1;
        double[] returns = new double[n];
        for (int i = 0; i < n; i++) {
            double prev = equityCurve.get(i);
            if (Math.abs(prev) < 1e-12) {
                returns[i] = 0.0;
            } else {
                returns[i] = (equityCurve.get(i + 1) - prev) / prev;
            }
        }

        double mean = 0.0;
        int finiteCount = 0;
        for (double r : returns) {
            if (Double.isFinite(r)) {
                mean += r;
                finiteCount++;
            }
        }
        if (finiteCount < 2) return 0.0;
        mean /= finiteCount;

        double variance = 0.0;
        for (double r : returns) {
            if (Double.isFinite(r)) {
                double d = r - mean;
                variance += d * d;
            }
        }
        variance /= finiteCount;
        double std = Math.sqrt(variance);

        if (std < 1e-12) return 0.0;
        // Annualize: sqrt(86400 * 252) ~ sqrt(21772800) ~ 4666.1
        return (mean / std) * Math.sqrt(86400.0 * 252.0);
    }

    private double[] computeMaxDrawdown() {
        double maxDd = 0.0;
        double maxDdPct = 0.0;
        if (equityCurve.isEmpty()) return new double[]{0.0, 0.0};

        double peak = equityCurve.get(0);
        for (double val : equityCurve) {
            if (val > peak) peak = val;
            double dd = peak - val;
            double ddPct = peak > 0 ? dd / peak : 0.0;
            if (dd > maxDd) maxDd = dd;
            if (ddPct > maxDdPct) maxDdPct = ddPct;
        }
        return new double[]{maxDd, maxDdPct};
    }

    // ── Accessors ─────────────────────────────────────────────────────

    public String getAgentName() { return agentName; }
    public double getInitialCash() { return initialCash; }
    public double getRealizedPnl() { return realizedPnl; }
    public List<Double> getEquityCurve() { return Collections.unmodifiableList(equityCurve); }
    public List<Map<String, Object>> getSnapshots() { return Collections.unmodifiableList(snapshots); }
    public int getTotalTrades() { return totalTrades; }
}
