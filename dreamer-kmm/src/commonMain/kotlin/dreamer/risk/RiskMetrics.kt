package dreamer.risk

data class RiskState(
    val peakTotalValue: Double = 0.0,
    val maxDrawdownPercent: Double = 0.0,
)

/**
 * Update risk metrics from current total portfolio value.
 * JS source (lines 717-722):
 *   if (currentTotalValue > this.peakTotalValue) {
 *     this.peakTotalValue = currentTotalValue;
 *   } else if (this.peakTotalValue > 0) {
 *     const drawdown = (this.peakTotalValue - currentTotalValue) / this.peakTotalValue;
 *     if (drawdown > this.maxDrawdownPercent) this.maxDrawdownPercent = drawdown;
 *   }
 */
fun updateRiskMetrics(currentTotalValue: Double, state: RiskState): RiskState {
    var peak = state.peakTotalValue
    var maxDD = state.maxDrawdownPercent
    if (currentTotalValue > peak) {
        peak = currentTotalValue
    } else if (peak > 0.0) {
        val drawdown = (peak - currentTotalValue) / peak
        if (drawdown > maxDD) maxDD = drawdown
    }
    return RiskState(peak, maxDD)
}
