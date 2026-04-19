/**
 * AssetInit — Auto-initialize new assets & detect manual cash extraction.
 *
 * Port of Dreamer 1.2.js lines 664–710.
 *
 * Pure functional, no I/O.  All inputs/outputs are [DoubleArray] for
 * autovectorization-friendly loops on Kotlin/Native and Kotlin/WASM.
 *
 * @see dreamer.tensor.PortfolioTensor
 */

package dreamer.engine

// ──────────────────────────────────────────────────────────────────────────────
// Result types
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Result of auto-initializing baselines for newly detected assets.
 *
 * @param stateChanged     `true` if at least one baseline was written.
 * @param updatedBaselines Copy of [DoubleArray] with new baselines applied.
 * @param skipTrading      Always `false` for auto-init (trading continues).
 */
data class AssetInitResult(
    val stateChanged: Boolean,
    val updatedBaselines: DoubleArray,
    val skipTrading: Boolean,
)

/**
 * Result of cash-extraction detection & baseline healing.
 *
 * @param extracted        `true` when a cash extraction event was detected
 *                         and baselines were healed.
 * @param healedBaselines  Updated baselines (healed or untouched copy).
 * @param skipTrading      `true` when extraction was detected — caller should
 *                         skip the trading cycle.
 */
data class CashExtractionResult(
    val extracted: Boolean,
    val healedBaselines: DoubleArray,
    val skipTrading: Boolean,
)

// ──────────────────────────────────────────────────────────────────────────────
// Auto-init new assets  (JS lines 664–675)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Initialize baselines for assets that have no baseline yet.
 *
 * For each asset: if `value > [minValue]` **and** the existing baseline is
 * missing / zero / negative, the baseline is set to the current value.
 *
 * Matches Dreamer 1.2.js lines 664–675:
 * ```
 * if (row.Value > 1.0 && (!baselines[sym] || baselines[sym] <= 0)) {
 *     baselines[sym] = row.Value;
 *     stateChanged = true;
 * }
 * ```
 *
 * @param symbols   Asset symbol labels (index-correlated with arrays).
 * @param values    Current USD value per asset.
 * @param baselines Existing baselines per asset.
 * @param minValue  Minimum value threshold to consider an asset "present"
 *                  (default 1.0, matching JS).
 * @return (stateChanged, updatedBaselines) pair.
 */
fun autoInitNewAssets(
    symbols: Array<String>,
    values: DoubleArray,
    baselines: DoubleArray,
    minValue: Double = 1.0,
): Pair<Boolean, DoubleArray> {
    val n = symbols.size
    require(values.size == n)     { "values.size ${values.size} != symbols.size $n" }
    require(baselines.size == n)  { "baselines.size ${baselines.size} != symbols.size $n" }

    val updated = baselines.copyOf()
    var stateChanged = false

    for (i in 0 until n) {
        if (values[i] > minValue && (updated[i] == 0.0 || updated[i] < 0.0)) {
            updated[i] = values[i]
            stateChanged = true
        }
    }

    return Pair(stateChanged, updated)
}

// ──────────────────────────────────────────────────────────────────────────────
// Cash extraction detection  (JS lines 678–710)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Detect manual cash extraction (user withdrew funds) and heal baselines.
 *
 * When total portfolio value drops sharply (>12 %) but asset prices are
 * stable (<2 % average change), we infer the drop came from a manual
 * withdrawal rather than market movement.  Baselines are healed to
 * `currentValue * healFactor` so the bot doesn't try to "rebalance"
 * against a phantom gap.
 *
 * Matches Dreamer 1.2.js lines 678–710:
 * ```
 * const dropPercent = (lastTotalValue - currentTotalValue) / lastTotalValue;
 * if (dropPercent > 0.12) {
 *     // compute avg price change across known symbols
 *     if (avgPriceChange < 0.02) {
 *         // heal baselines to currentValue * 0.995
 *         return skip trading
 *     }
 * }
 * ```
 *
 * @param lastTotalValue       Total portfolio value from the previous cycle.
 * @param currentTotalValue    Total portfolio value this cycle.
 * @param symbols              Asset symbol labels.
 * @param currentPrices        Current price per asset.
 * @param lastPrices           Price per asset from the previous cycle.
 *                             A value of `0.0` is treated as "no previous price"
 *                             (mirrors the JS falsy check `lastCyclePrices[sym]`).
 * @param currentValues        Current USD value per asset.
 * @param baselines            Existing baselines per asset.
 * @param dropThreshold        Fractional drop to trigger detection (default 0.12 = 12 %).
 * @param priceStabilityThreshold  Max avg price change to confirm extraction (default 0.02 = 2 %).
 * @param healFactor           Multiplier on currentValue for healed baseline (default 0.995).
 * @return [CashExtractionResult] with healed baselines and skip flag.
 */
fun detectCashExtraction(
    lastTotalValue: Double,
    currentTotalValue: Double,
    symbols: Array<String>,
    currentPrices: DoubleArray,
    lastPrices: DoubleArray,
    currentValues: DoubleArray,
    baselines: DoubleArray,
    dropThreshold: Double = 0.12,
    priceStabilityThreshold: Double = 0.02,
    healFactor: Double = 0.995,
): CashExtractionResult {
    val n = symbols.size
    require(currentPrices.size == n)  { "currentPrices.size != $n" }
    require(lastPrices.size == n)     { "lastPrices.size != $n" }
    require(currentValues.size == n)  { "currentValues.size != $n" }
    require(baselines.size == n)      { "baselines.size != $n" }

    // Guard: need a valid previous total to compute a meaningful drop
    if (lastTotalValue <= 0.0) {
        return CashExtractionResult(
            extracted = false,
            healedBaselines = baselines.copyOf(),
            skipTrading = false,
        )
    }

    val dropPercent = (lastTotalValue - currentTotalValue) / lastTotalValue

    if (dropPercent > dropThreshold) {
        // Compute average absolute price change across assets with a known lastPrice
        var totalPriceChangePct = 0.0
        var priceCount = 0

        for (i in 0 until n) {
            val prevP = lastPrices[i]
            if (prevP > 0.0) {
                val change = kotlin.math.abs((currentPrices[i] - prevP) / prevP)
                totalPriceChangePct += change
                priceCount++
            }
        }

        val avgPriceChange = if (priceCount > 0) totalPriceChangePct / priceCount else 0.0

        if (avgPriceChange < priceStabilityThreshold) {
            // Extraction detected — heal baselines
            val healed = DoubleArray(n)
            for (i in 0 until n) {
                healed[i] = if (baselines[i] > 0.0) {
                    currentValues[i] * healFactor
                } else {
                    baselines[i]
                }
            }
            return CashExtractionResult(
                extracted = true,
                healedBaselines = healed,
                skipTrading = true,
            )
        }
    }

    return CashExtractionResult(
        extracted = false,
        healedBaselines = baselines.copyOf(),
        skipTrading = false,
    )
}
