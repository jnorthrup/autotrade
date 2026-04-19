@file:OptIn(kotlin.js.ExperimentalJsExport::class)

/**
 * JS (Node.js) exported Dreamer kernels — thin @JsExport wrappers that
 * accept/return JS-native Array<Double> and delegate to the commonMain
 * DoubleArrayMath / signal / regime functions.
 *
 * These functions are compiled into the JS IR module and callable from
 * Node.js as module-level exports.
 */

package dreamer.platform

import dreamer.tensor.ema
import dreamer.tensor.sparDrag
import dreamer.tensor.portfolioDeviation
import dreamer.tensor.mean
import dreamer.tensor.variance
import dreamer.signal.crashProtectionActive
import dreamer.regime.classify
import dreamer.exchange.roundQty
import dreamer.exchange.checkMinTrade
import dreamer.exchange.checkMinQuantity

// ──────────────────────────────────────────────────────────────────────────────
// Helpers: JsArray<Number> / Array<Double> <-> DoubleArray conversion
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Convert a JS Array of numbers into a Kotlin [DoubleArray].
 *
 * Accepts both [Array<Double>] and the JS-native representation that
 * Kotlin/JS IR uses for typed interop.
 */
private fun Array<Double>.toDoubleArray(): DoubleArray {
    val n = this.size
    val out = DoubleArray(n)
    for (i in 0 until n) {
        out[i] = this[i]
    }
    return out
}

/**
 * Convert a Kotlin [DoubleArray] into a JS-compatible [Array<Double>].
 */
private fun DoubleArray.toArray(): Array<Double> {
    val n = this.size
    val out = Array<Double>(n) { 0.0 }
    for (i in 0 until n) {
        out[i] = this[i]
    }
    return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Exported kernels
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Exponential Moving Average.
 *
 * @param history  Price history as a JS Array of doubles.
 * @param span     EMA span (>= 1).
 * @return EMA result as a new JS Array of doubles.
 */
@JsExport
fun dreamerEma(history: Array<Double>, span: Int): Array<Double> {
    return ema(history.toDoubleArray(), span).toArray()
}

/**
 * Spar drag — asymmetric drag pulling [current] toward [baseline].
 *
 * @param baseline    Baseline values.
 * @param current     Current values (same length).
 * @param coefficient Drag coefficient.
 * @return Dragged values as a new JS Array of doubles.
 */
@JsExport
fun dreamerSparDrag(baseline: Array<Double>, current: Array<Double>, coefficient: Double): Array<Double> {
    return sparDrag(baseline.toDoubleArray(), current.toDoubleArray(), coefficient).toArray()
}

/**
 * Root-mean-square deviation of [values] from [baselines].
 *
 * @param values    Current values.
 * @param baselines Baseline values (same length).
 * @return RMS deviation as a Double.
 */
@JsExport
fun dreamerPortfolioDeviation(values: Array<Double>, baselines: Array<Double>): Double {
    return portfolioDeviation(values.toDoubleArray(), baselines.toDoubleArray())
}

/**
 * Determine whether portfolio-wide crash protection is active.
 *
 * @param values              Current portfolio values per asset.
 * @param baselines           Baseline values per asset (same length).
 * @param triggerMinNegDevPct Negative-deviation threshold (e.g. -0.07).
 * @param triggerAssetPct     Fraction of assets that must be declining (e.g. 0.70).
 * @return true when crash protection should be engaged.
 */
@JsExport
fun dreamerCrashProtection(
    values: Array<Double>,
    baselines: Array<Double>,
    triggerMinNegDevPct: Double,
    triggerAssetPct: Double
): Boolean {
    return crashProtectionActive(
        values.toDoubleArray(),
        baselines.toDoubleArray(),
        triggerMinNegDevPct,
        triggerAssetPct
    )
}

/**
 * Arithmetic mean.
 *
 * @param values Input values as a JS Array of doubles.
 * @return Mean, or NaN for an empty array.
 */
@JsExport
fun dreamerMean(values: Array<Double>): Double {
    return mean(values.toDoubleArray())
}

/**
 * Population variance.
 *
 * @param values Input values as a JS Array of doubles.
 * @return Variance, or NaN for an empty array.
 */
@JsExport
fun dreamerVariance(values: Array<Double>): Double {
    return variance(values.toDoubleArray())
}

/**
 * Classify the market regime from a price history.
 *
 * @param history Price history as a JS Array of doubles.
 * @return Regime code integer (0=UNKNOWN, 1=CRAB_CHOP, 2=BULL_RUSH,
 *         3=BEAR_CRASH, 4=STEADY_GROWTH, 5=VOLATILE_CHOP).
 */
@JsExport
fun dreamerClassifyRegime(history: Array<Double>): Int {
    return classify(history.toDoubleArray()).code
}

// ── Exchange helpers ──────────────────────────────────────────────────────

/**
 * Round quantity down to nearest valid increment.
 * @param symbol    Asset symbol (e.g. "BTC").
 * @param qty       Raw quantity.
 * @param minIncr   Minimum increment step (e.g. 0.00000001).
 * @return Rounded quantity as a string.
 */
@JsExport
fun dreamerRoundQty(symbol: String, qty: Double, minIncr: Double): String {
    return roundQty(symbol, qty, mapOf(symbol to minIncr))
}

/**
 * Check if a USD trade value exceeds minimum dust threshold ($0.25).
 */
@JsExport
fun dreamerCheckMinTrade(usdValue: Double): Boolean = checkMinTrade(usdValue)

/**
 * Check if a quantity meets minimum order size for a symbol.
 * @param symbol  Asset symbol.
 * @param qty     Order quantity.
 * @param minQty  Minimum order quantity.
 */
@JsExport
fun dreamerCheckMinQuantity(symbol: String, qty: Double, minQty: Double): Boolean {
    return checkMinQuantity(symbol, qty, mapOf(symbol to minQty))
}
