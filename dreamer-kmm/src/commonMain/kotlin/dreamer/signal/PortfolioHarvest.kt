/**
 * PortfolioHarvest — pure-functional portfolio-level harvest evaluator.
 *
 * Port of Dreamer 1.2.js lines 814-888 (Portfolio Override Harvest Logic)
 * to Kotlin Multiplatform commonMain.  All parameters are passed in; no I/O
 * or side effects occur.  The caller (TradingEngine) is responsible for
 * executing any sell instructions returned.
 *
 * Usage:
 * ```
 *   val result = evaluatePortfolioHarvest(
 *       portfolioDeviationPercent = ...,
 *       assets = assetRows,
 *       excludeSymbols = harvestExclude,
 *       nowMs = nowMs,
 *       state = currentState,
 *       params = portfolioHarvestParams,
 *   )
 *   val newState = result.updatedState
 *   if (result.executed) { /* execute result.sells */ }
 * ```
 */
package dreamer.signal

import kotlin.math.max

// ── Constants ────────────────────────────────────────────────────────────────

/** Hysteresis band around previousDeviation for cycle-counting decisions. */
private const val PRECISION_THRESHOLD = 0.0001

// ── Data classes ─────────────────────────────────────────────────────────────

/**
 * Portfolio-level harvest state carried across evaluation cycles.
 *
 * Mirrors the JS `portfolioHarvestState` object:
 *   - `flagged`                — whether portfolio deviation is above trigger.
 *   - `cycleCount`             — number of confirming deviation-improvement ticks.
 *   - `previousDeviationPercent` — last cycle's portfolio deviation (null before first tick).
 *   - `flaggedAt`              — epoch-millis when the flag was first raised.
 */
data class PortfolioHarvestState(
    val flagged: Boolean = false,
    val cycleCount: Int = 0,
    val previousDeviationPercent: Double? = null,
    val flaggedAt: Long? = null,
)

/**
 * Per-asset sell instruction produced by a portfolio harvest evaluation.
 *
 * @param symbolIndex Position of the asset in the input list (for caller bookkeeping).
 * @param symbol      Trading pair symbol (e.g. "BTC").
 * @param qtyToSell   Quantity of the asset to sell (USD-denominated surplus / price).
 * @param surplusUSD  `value - baseline` for this asset at the time of the decision.
 */
data class PortfolioSellInstruction(
    val symbolIndex: Int,
    val symbol: String,
    val qtyToSell: Double,
    val surplusUSD: Double,
)

/**
 * Immutable result of a single [evaluatePortfolioHarvest] call.
 *
 * @param executed      true when the confirmation threshold was met and sell
 *                      instructions were generated.
 * @param sells         Sell instructions for each qualifying asset (empty when
 *                      [executed] is false).
 * @param updatedState  The [PortfolioHarvestState] to store for the next cycle.
 *                      Resets to unflagged after a successful execution.
 */
data class PortfolioHarvestResult(
    val executed: Boolean,
    val sells: List<PortfolioSellInstruction>,
    val updatedState: PortfolioHarvestState,
)

// ── Asset input row ──────────────────────────────────────────────────────────

/**
 * Read-only snapshot of a single asset's portfolio data.
 *
 * @param symbol   Trading pair symbol (e.g. "BTC").
 * @param baseline Current baseline value for this asset in USD.
 * @param value    Current market value of the holding in USD.
 * @param price    Current price per unit of the asset in USD.
 */
data class PortfolioAssetRow(
    val symbol: String,
    val baseline: Double,
    val value: Double,
    val price: Double,
)

// ── Parameters bundle ────────────────────────────────────────────────────────

/**
 * Resolved genome parameters needed by the portfolio harvest evaluator.
 *
 * Callers extract these from a [dreamer.genome.Genome] once per evaluation
 * cycle and pass the bundle in so the pure function has zero coupling to
 * the Genome class.
 *
 * Ordinal mapping:
 *   - [triggerDeviationPercent]  ← Genome ordinal 7  (PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT)
 *   - [confirmationCycles]       ← Genome ordinal 8  (PORTFOLIO_HARVEST_CONFIRMATION_CYCLES)
 *   - [minAssetSurplus]          ← Genome ordinal 9  (MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST)
 */
data class PortfolioHarvestParams(
    val triggerDeviationPercent: Double,   // genome value (e.g. 0.035); multiplied by 100 internally
    val confirmationCycles: Double,        // integral value (e.g. 3.0)
    val minAssetSurplus: Double,           // minimum USD surplus per asset to qualify for sell
)

// ── Pure evaluation function ─────────────────────────────────────────────────

/**
 * Evaluate whether a portfolio-level baseline reset harvest should occur.
 *
 * Algorithm (mirrors Dreamer 1.2.js lines 814-888):
 *
 *  1. If [enablePortfolioHarvest] is false, return immediately (no-op, state unchanged).
 *  2. Compute `triggerValue = triggerDeviationPercent * 100` (genome 0.035 → trigger 3.5).
 *  3. **Flag logic**:
 *      - If not flagged and [portfolioDeviationPercent] >= triggerValue → flag,
 *        reset counters, record `flaggedAt = nowMs`.
 *      - If flagged and [portfolioDeviationPercent] < triggerValue → unflag
 *        (reset state to defaults).
 *  4. **Cycle counting** (only while flagged):
 *      - `currDev < prevDev - PRECISION_THRESHOLD` → cycleCount++.
 *      - `currDev > prevDev + PRECISION_THRESHOLD` → cycleCount = max(0, cycleCount - 1).
 *      - Update `previousDeviationPercent` to current deviation.
 *  5. **Execution trigger**: flagged AND cycleCount >= confirmationCycles.
 *  6. If triggered: for each asset where `value > baseline` and
 *     `surplus >= minAssetSurplus`, compute `qtyToSell = surplusUSD / price`
 *     and produce a [PortfolioSellInstruction].
 *  7. After execution, reset state to unflagged.
 *
 * @param enablePortfolioHarvest  Master switch from Genome.enablePortfolioHarvest.
 * @param portfolioDeviationPercent  Current portfolio deviation as a percent
 *        (e.g. 3.5 means 3.5% deviation).
 * @param assets  List of [PortfolioAssetRow] snapshots for each tracked asset.
 * @param excludeSymbols  Set of symbols to skip (mirrors JS `HARVEST_EXCLUDE`).
 * @param nowMs  Current epoch-millis (used for `flaggedAt` timestamp).
 * @param state  Prior [PortfolioHarvestState] from the previous cycle.
 * @param params Resolved [PortfolioHarvestParams] from the genome.
 * @return [PortfolioHarvestResult] with sell instructions and updated state.
 */
fun evaluatePortfolioHarvest(
    enablePortfolioHarvest: Boolean,
    portfolioDeviationPercent: Double,
    assets: List<PortfolioAssetRow>,
    excludeSymbols: Set<String>,
    nowMs: Long,
    state: PortfolioHarvestState,
    params: PortfolioHarvestParams,
): PortfolioHarvestResult {

    // ── Early exit: feature disabled ─────────────────────────────────────

    if (!enablePortfolioHarvest) {
        return PortfolioHarvestResult(
            executed = false,
            sells = emptyList(),
            updatedState = state,
        )
    }

    val triggerValue = params.triggerDeviationPercent * 100.0

    // ── Flag management ──────────────────────────────────────────────────

    var flagged = state.flagged
    var cycleCount = state.cycleCount
    var previousDev = state.previousDeviationPercent
    var flaggedAt = state.flaggedAt

    if (!flagged && portfolioDeviationPercent >= triggerValue) {
        // Raise the flag — start tracking.
        flagged = true
        cycleCount = 0
        previousDev = portfolioDeviationPercent
        flaggedAt = nowMs
    } else if (flagged && portfolioDeviationPercent < triggerValue) {
        // Dropped below trigger — clear flag.
        flagged = false
        cycleCount = 0
        previousDev = null
        flaggedAt = null
    }

    // ── Cycle counting (only while flagged) ───────────────────────────────

    if (flagged && previousDev != null) {
        val currDev = portfolioDeviationPercent
        when {
            currDev < previousDev - PRECISION_THRESHOLD -> cycleCount++
            currDev > previousDev + PRECISION_THRESHOLD ->
                cycleCount = max(0, cycleCount - 1)
        }
        // Always update previous deviation to current for next cycle.
        previousDev = portfolioDeviationPercent
    } else if (flagged) {
        // flagged but previousDev is null (just flagged this cycle) — set it.
        previousDev = portfolioDeviationPercent
    }

    // Build intermediate state for the cycle-counting step.
    val trackedState = PortfolioHarvestState(
        flagged = flagged,
        cycleCount = cycleCount,
        previousDeviationPercent = previousDev,
        flaggedAt = flaggedAt,
    )

    // ── Execution trigger check ──────────────────────────────────────────

    val confirmationCycles = params.confirmationCycles.toInt()

    if (!flagged || cycleCount < confirmationCycles) {
        return PortfolioHarvestResult(
            executed = false,
            sells = emptyList(),
            updatedState = trackedState,
        )
    }

    // ── Build sell instructions ──────────────────────────────────────────

    val sells = mutableListOf<PortfolioSellInstruction>()

    for (i in assets.indices) {
        val row = assets[i]

        // Skip excluded symbols.
        if (row.symbol in excludeSymbols) continue
        // Skip assets with no baseline.
        if (row.baseline <= 0.0) continue
        // Only harvest assets above their baseline.
        if (row.value <= row.baseline) continue

        val surplusUSD = row.value - row.baseline

        // Skip assets below the minimum surplus threshold.
        if (surplusUSD < params.minAssetSurplus) continue

        val qtyToSell = if (row.price > 0.0) surplusUSD / row.price else 0.0

        if (qtyToSell > 0.0) {
            sells.add(
                PortfolioSellInstruction(
                    symbolIndex = i,
                    symbol = row.symbol,
                    qtyToSell = qtyToSell,
                    surplusUSD = surplusUSD,
                )
            )
        }
    }

    // After execution, reset state to unflagged (mirrors JS line 886).
    val resetState = PortfolioHarvestState(
        flagged = false,
        cycleCount = 0,
        previousDeviationPercent = null,
        flaggedAt = null,
    )

    return PortfolioHarvestResult(
        executed = true,
        sells = sells,
        updatedState = resetState,
    )
}
