/**
 * Crash-protection signal kernels for dreamer-kmm commonMain.
 *
 * Pure functions over [DoubleArray] / [IntArray] with plain for-loops —
 * no JVM-specific APIs, friendly to LLVM / WASM auto-vectorization.
 *
 * Logic mirrors Dreamer 1.2.js crash protection (lines 794–812):
 *   - Count assets with a positive baseline.
 *   - For each such asset, compute deviation = (value − baseline) / baseline.
 *   - Count those whose deviation is ≤ the negative-deviation trigger threshold.
 *   - If the fraction of declining assets ≥ the asset-percent trigger ⇒ crash
 *     protection is active.
 */

package dreamer.signal

// ──────────────────────────────────────────────────────────────────────────────
// Crash protection
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Determines whether the portfolio-wide crash-protection signal is active.
 *
 * @param values     Current portfolio values per asset (e.g. USD value).
 * @param baselines  Baseline (target) values per asset, same length as [values].
 * @param triggerMinNegDevPercent  Negative-deviation threshold (e.g. −0.07).
 *         An asset is "declining" when its deviation is **≤** this value.
 * @param triggerAssetPercent  Fraction of baseline-tracked assets that must be
 *         declining to activate crash protection (e.g. 0.70).
 * @return `true` when crash protection should be engaged.
 */
fun crashProtectionActive(
    values: DoubleArray,
    baselines: DoubleArray,
    triggerMinNegDevPercent: Double,
    triggerAssetPercent: Double
): Boolean {
    require(values.size == baselines.size) {
        "values.size=${values.size} != baselines.size=${baselines.size}"
    }

    var totalWithBaseline = 0
    var decliningCount = 0

    for (i in values.indices) {
        val baseline = baselines[i]
        if (baseline > 0.0) {
            totalWithBaseline++
            val deviation = (values[i] - baseline) / baseline
            if (deviation <= triggerMinNegDevPercent) {
                decliningCount++
            }
        }
    }

    if (totalWithBaseline == 0) return false

    val decliningFraction = decliningCount.toDouble() / totalWithBaseline.toDouble()
    return decliningFraction >= triggerAssetPercent
}

// ──────────────────────────────────────────────────────────────────────────────
// Portfolio deviation
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Portfolio-wide deviation percentage — matches Dreamer 1.2.js exactly.
 *
 * JS formula (line 736-737):
 * ```
 * sum(value_i - baseline_i) / sum(baseline_i) * 100
 * ```
 * for assets with baseline > 0, excluding [excludeIndices].
 *
 * This is a *baseline-weighted* deviation — assets with larger baselines
 * contribute proportionally more. Different from unweighted mean deviation.
 *
 * @param values          Current portfolio values per asset.
 * @param baselines       Baseline (target) values per asset, same length as [values].
 * @param excludeIndices  Asset indices to skip (e.g. already harvested). Empty by default.
 * @return deviation in percent (e.g. 3.5 means 3.5%), or NaN if no qualifying assets.
 */
fun portfolioDeviationPercent(
    values: DoubleArray,
    baselines: DoubleArray,
    excludeIndices: IntArray = IntArray(0)
): Double {
    require(values.size == baselines.size) {
        "values.size=${values.size} != baselines.size=${baselines.size}"
    }

    var totalDifference = 0.0
    var totalBaseline = 0.0

    for (i in values.indices) {
        // Skip excluded indices
        var excluded = false
        for (j in excludeIndices.indices) {
            if (excludeIndices[j] == i) {
                excluded = true
                break
            }
        }
        if (excluded) continue

        val baseline = baselines[i]
        if (baseline > 0.0) {
            totalDifference += (values[i] - baseline)
            totalBaseline += baseline
        }
    }

    if (totalBaseline <= 0.0) return Double.NaN
    return (totalDifference / totalBaseline) * 100.0
}
