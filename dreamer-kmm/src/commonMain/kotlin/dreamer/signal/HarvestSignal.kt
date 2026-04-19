/**
 * HarvestSignal — pure-functional individual-asset harvest evaluator.
 *
 * Port of Dreamer 1.2.js individual harvest logic (lines 890-994) to Kotlin
 * Multiplatform commonMain.  All parameters are read from the [Genome]
 * ordinal array; no I/O or side effects occur.
 *
 * Usage:
 * ```
 *   val result = evaluateHarvest(totalValue, baseline, price, nowMs, state, params)
 *   val newState = result.updatedState   // feed back into trailing-state map
 * ```
 */
package dreamer.signal

import kotlin.math.max

// ── Constants ────────────────────────────────────────────────────────────────

/** Hysteresis band around previousDeviation for cycle-counting decisions. */
private const val PRECISION_THRESHOLD = 0.0001

// ── Harvest type ─────────────────────────────────────────────────────────────

/**
 * Classification of the harvest decision returned by [evaluateHarvest].
 *
 * - [NONE]    — no harvest this cycle.
 * - [STANDARD] — triggered by cycle-count confirmation.
 * - [FORCED]   — triggered by timeout while flagged.
 */
enum class HarvestType {
    NONE,
    STANDARD,
    FORCED,
}

// ── Data classes ─────────────────────────────────────────────────────────────

/**
 * Per-asset trailing state carried across evaluation cycles.
 *
 * Mirrors the JS `trailingState[sym]` object:
 *   - `flagged`           — whether the asset is above its trigger threshold.
 *   - `harvestCycleCount` — number of confirming deviation-improvement ticks.
 *   - `flaggedAt`         — epoch-millis when the flag was first raised.
 *   - `previousDeviation` — last cycle's deviation (null before first tick).
 */
data class TrailingState(
    val flagged: Boolean = false,
    val harvestCycleCount: Int = 0,
    val flaggedAt: Long = 0L,
    val previousDeviation: Double? = null,
)

/**
 * Immutable result of a single [evaluateHarvest] call.
 *
 * @param type         [HarvestType.NONE] if nothing to do, otherwise the
 *                     trigger that fired.
 * @param qtyToSell    Quantity of the asset to sell (0.0 when [NONE]).
 * @param surplus      `totalValue - baseline` at the time of the decision.
 * @param newBaseline  Updated baseline after harvest (unchanged when [NONE]).
 * @param updatedState The [TrailingState] to store for the next cycle.
 *                     On a successful harvest this resets to unflagged with
 *                     zeroed counters.  On a failed surplus check it may
 *                     reset counters or clear the flag depending on type.
 */
data class HarvestResult(
    val type: HarvestType,
    val qtyToSell: Double,
    val surplus: Double,
    val newBaseline: Double,
    val updatedState: TrailingState,
) {
    companion object {
        /** Convenience: no harvest, state unchanged except for deviation tracking. */
        fun none(baseline: Double, state: TrailingState): HarvestResult =
            HarvestResult(
                type = HarvestType.NONE,
                qtyToSell = 0.0,
                surplus = 0.0,
                newBaseline = baseline,
                updatedState = state,
            )
    }
}

// ── Parameters bundle ────────────────────────────────────────────────────────

/**
 * Resolved genome parameters needed by the harvest evaluator.
 *
 * Callers extract these from a [Genome] once per evaluation cycle (typically
 * via `Genome.resolve(asset)` or `Genome.d(ordinal, asset)`) and pass the
 * bundle in so the pure function has zero coupling to the Genome class.
 */
data class HarvestParams(
    val triggerPercent: Double,         // FLAT_HARVEST_TRIGGER_PERCENT
    val takePercent: Double,            // HARVEST_TAKE_PERCENT
    val cycleThreshold: Double,         // HARVEST_CYCLE_THRESHOLD  (integral value)
    val minSurplus: Double,             // MIN_SURPLUS_FOR_HARVEST
    val minSurplusForced: Double,       // MIN_SURPLUS_FOR_FORCED_HARVEST
    val forcedTimeoutMs: Double,        // FORCED_HARVEST_TIMEOUT_MS
    val targetAdjustPercent: Double,    // TARGET_ADJUST_PERCENT
)

// ── Pure evaluation function ─────────────────────────────────────────────────

/**
 * Evaluate whether an individual asset should be harvested.
 *
 * Algorithm (mirrors Dreamer 1.2.js lines 890-994):
 *
 *  1. Compute `currentDeviation = (totalValue - baseline) / baseline`.
 *  2. Compute `triggerValue = baseline * (1 + triggerPercent)`.
 *  3. **Flag logic**:
 *      - If not flagged and `totalValue >= triggerValue` → flag, reset counters,
 *        record `flaggedAt = nowMs`.
 *      - If flagged and `totalValue < triggerValue` → unflag (reset state).
 *      - If not flagged → return [HarvestType.NONE].
 *  4. **Cycle counting** (only while flagged):
 *      - `currentDeviation < previousDeviation - PRECISION_THRESHOLD` → increment.
 *      - `currentDeviation > previousDeviation + PRECISION_THRESHOLD` → decrement (min 0).
 *  5. **Forced harvest**: `flaggedDuration > forcedTimeoutMs`.
 *  6. **Standard harvest**: `harvestCycleCount >= cycleThreshold`.
 *  7. If triggered, check `surplus >= minSurplus` (different floor for forced vs standard).
 *  8. `qtyToSell = surplus * takePercent / price`.
 *  9. `newBaseline = (baseline + retainedSurplus) * (1 + targetAdjustPercent)`.
 *
 * @param totalValue  Current USD value of the asset holding.
 * @param baseline    Current baseline for this asset (must be > 0).
 * @param price       Current asset price (must be > 0 for harvest).
 * @param nowMs       Current epoch-millis (used for forced-timeout check).
 * @param state       Prior [TrailingState] for this asset.
 * @param params      Resolved [HarvestParams] from the genome.
 * @return [HarvestResult] with the decision and updated trailing state.
 */
fun evaluateHarvest(
    totalValue: Double,
    baseline: Double,
    price: Double,
    nowMs: Long,
    state: TrailingState,
    params: HarvestParams,
): HarvestResult {

    // Guard: invalid baseline
    if (baseline <= 0.0) return HarvestResult.none(baseline, state)

    val currentDeviation = (totalValue - baseline) / baseline
    val triggerValue = baseline * (1.0 + params.triggerPercent)

    // ── Flag management ──────────────────────────────────────────────────

    if (!state.flagged) {
        // Not currently flagged — raise the flag if we cross the trigger.
        return if (totalValue >= triggerValue) {
            val newState = TrailingState(
                flagged = true,
                harvestCycleCount = 0,
                flaggedAt = nowMs,
                previousDeviation = currentDeviation,
            )
            HarvestResult.none(baseline, newState)
        } else {
            HarvestResult.none(baseline, state)
        }
    }

    // Currently flagged — drop the flag if we fall back below trigger.
    if (totalValue < triggerValue) {
        return HarvestResult.none(baseline, TrailingState())
    }

    // ── Cycle counting ───────────────────────────────────────────────────

    var cycleCount = state.harvestCycleCount
    val prevDev = state.previousDeviation
    if (prevDev != null) {
        when {
            currentDeviation < prevDev - PRECISION_THRESHOLD -> cycleCount++
            currentDeviation > prevDev + PRECISION_THRESHOLD ->
                cycleCount = max(0, cycleCount - 1)
        }
    }

    // Update state with new deviation and cycle count for next call.
    val trackedState = state.copy(
        harvestCycleCount = cycleCount,
        previousDeviation = currentDeviation,
    )

    // ── Harvest trigger check ────────────────────────────────────────────

    val flaggedDuration = nowMs - state.flaggedAt

    val isForced = flaggedDuration > params.forcedTimeoutMs
    val isStandard = cycleCount >= params.cycleThreshold.toInt()

    if (!isForced && !isStandard) {
        return HarvestResult.none(baseline, trackedState)
    }

    // ── Surplus check ────────────────────────────────────────────────────

    val surplus = totalValue - baseline
    val harvestType = if (isForced) HarvestType.FORCED else HarvestType.STANDARD
    val minSurplus = if (isForced) params.minSurplusForced else params.minSurplus

    if (surplus < minSurplus) {
        // JS behavior: surplus too small → no state reset. Trailing state keeps
        // accumulating cycles toward a future harvest. Matches Dreamer 1.2.js
        // lines 933-937 where the if(surplus >= minSurplus) block is simply skipped.
        return HarvestResult(
            type = HarvestType.NONE,
            qtyToSell = 0.0,
            surplus = surplus,
            newBaseline = baseline,
            updatedState = trackedState,
        )
    }

    // ── Execute harvest (compute outputs) ────────────────────────────────

    val qtyToSell = if (price > 0.0) (surplus * params.takePercent) / price else 0.0
    val retainedSurplus = surplus * (1.0 - params.takePercent)
    val newBaseline = (baseline + retainedSurplus) * (1.0 + params.targetAdjustPercent)

    // After a successful harvest the trailing state resets to unflagged.
    return HarvestResult(
        type = harvestType,
        qtyToSell = qtyToSell,
        surplus = surplus,
        newBaseline = newBaseline,
        updatedState = TrailingState(),
    )
}
