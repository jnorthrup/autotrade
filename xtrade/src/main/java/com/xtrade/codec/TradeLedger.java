package com.xtrade.codec;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Trade performance ledger mirroring the Python BaseExpert.trade_ledger.
 */
public class TradeLedger {
    private double sharpe;
    private double hitRate;
    private double cumulativePnl;
    private int signalCount;

    public TradeLedger() {
        this.sharpe = 0.0;
        this.hitRate = 0.0;
        this.cumulativePnl = 0.0;
        this.signalCount = 0;
    }

    public void recordTradeOutcome(double realizedPnl, double direction, double actualReturn) {
        cumulativePnl += realizedPnl;
        signalCount++;
        if (realizedPnl > 0) {
            hitRate = (hitRate * (signalCount - 1) + 1.0) / signalCount;
        } else {
            hitRate = (hitRate * (signalCount - 1)) / signalCount;
        }
    }

    public Map<String, Object> getLedger() {
        Map<String, Object> ledger = new LinkedHashMap<>();
        ledger.put("sharpe", sharpe);
        ledger.put("hit_rate", hitRate);
        ledger.put("cumulative_pnl", cumulativePnl);
        ledger.put("signal_count", signalCount);
        if (signalCount >= 10) {
            ledger.put("sharpe", cumulativePnl / Math.max(1.0, signalCount));
        }
        return ledger;
    }

    public void reset() {
        this.sharpe = 0.0;
        this.hitRate = 0.0;
        this.cumulativePnl = 0.0;
        this.signalCount = 0;
    }

    public double getSharpe() { return sharpe; }
    public double getHitRate() { return hitRate; }
    public double getCumulativePnl() { return cumulativePnl; }
    public int getSignalCount() { return signalCount; }
}
