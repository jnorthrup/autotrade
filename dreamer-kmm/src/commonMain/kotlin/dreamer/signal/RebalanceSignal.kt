/**
 * RebalanceSignal -- pure-functional rebalance evaluation for Dreamer-KMM.
 *
 * Port of Dreamer 1.2.js rebalance logic (lines 1181-1328) to Kotlin
 * Multiplatform commonMain.  All decisions are deterministic from the
 * inputs; no I/O, no side effects, no time calls.
 *
 * Algorithm summary:
 *   1. Zero-balance guard: value < $1 -> drop state, NONE.
 *   2. Recovery check: value >= trigger threshold (with hysteresis) -> clear state, NONE.
 *   3. Trigger entry: value below threshold creates new [RebalanceState].
 *   4. Forced rebalance: duration > timeout -> buy shortfall * forcePercent.
 *   5. Positive-cycle tracking: deviation closer to 0 increments, further negative decrements.
 *   6. Standard rebalance: posCycleCount >= threshold, not on cooldown -> partial recovery buy.
 *   7. Post-standard: increment attemptCount; clear state if recovered, else cooldown at max attempts.
 */

package dreamer.signal

import kotlin.math.max

// ── Result type ─────────────────────────────────────────────────────────────

/**
 * Classification of the rebalance action to take.
 */
enum class RebalanceType {
    /** No action required this cycle. */
    NONE,
    /** Standard partial-recovery buy after sufficient positive deviation cycles. */
    STANDARD,
    /** Forced buy after the trigger timeout has elapsed. */
    FORCED,
}

// ── State ───────────────────────────────────────────────────────────────────

/**
 * Per-asset mutable rebalance tracking state.
 *
 * Mirrors the JS `rebalanceState[sym]` object.  Immutable value class --
 * the caller is responsible for storing the updated instance returned in
 * [RebalanceResult.updatedState].
 *
 * @property triggered            Always `true` when a state object exists.
 * @property triggeredAtMs        Epoch-millis when the trigger first fired.
 * @property posCycleCount        Count of consecutive positive deviation moves.
 * @property attemptCount         Number of standard rebalance attempts so far.
 * @property cooldownUntilMs      Epoch-millis before which no standard attempt.
 * @property baselineWhenTriggered  The baseline value at the moment of trigger.
 * @property previousDeviation    Last observed deviation ratio, or `null` if
 *                                not yet measured (first cycle after trigger or
 *                                reset after an attempt).
 */
data class RebalanceState(
    val triggered: Boolean = true,
    val triggeredAtMs: Long = 0L,
    val posCycleCount: Int = 0,
    val attemptCount: Int = 0,
    val cooldownUntilMs: Long = 0L,
    val baselineWhenTriggered: Double = 0.0,
    val previousDeviation: Double? = null,
)

// ── Config ──────────────────────────────────────────────────────────────────

/**
 * Genome-derived parameters driving rebalance decisions.
 *
 * Defaults match [Genome.DEFAULT_DOUBLES] ordinals 0, 21-29.
 * Construct via [RebalanceConfig.fromGenome] or use the zero-arg default.
 */
data class RebalanceConfig(
    val triggerPercent: Double = 0.035,
    val partialRecoveryPercent: Double = 0.70,
    val positiveThreshold: Int = 3,
    val maxAttempts: Int = 3,
    val cooldownMs: Long = 30L * 60L * 1000L,
    val forceTimeoutMs: Long = 25L * 60L * 1000L,
    val forceShortfallPercent: Double = 0.25,
    val minPartialRebalanceUSD: Double = 0.25,
    val minForcedRebalanceUSD: Double = 0.25,
    val targetAdjustPercent: Double = 0.001,
) {
    companion object {
        /**
         * Extract rebalance parameters from a resolved (per-asset) genome
         * DoubleArray using [Genome] companion ordinals.
         *
         * Pass `genome.resolve(symbol)` or `genome.doubles` for global defaults.
         */
        fun fromGenome(d: DoubleArray): RebalanceConfig = RebalanceConfig(
            triggerPercent       = d[21],  // FLAT_REBALANCE_TRIGGER_PERCENT
            partialRecoveryPercent = d[22],  // PARTIAL_RECOVERY_PERCENT
            positiveThreshold    = d[23].toInt(),  // REBALANCE_POSITIVE_THRESHOLD
            maxAttempts          = d[24].toInt(),  // MAX_REBALANCE_ATTEMPTS
            cooldownMs           = d[25].toLong(), // REBALANCE_COOLDOWN_MS
            forceTimeoutMs       = d[26].toLong(), // FORCE_REBALANCE_TIMEOUT_MS
            forceShortfallPercent = d[27],  // FORCE_REBALANCE_SHORTFALL_PERCENT
            minPartialRebalanceUSD = d[28], // MIN_PARTIAL_REBALANCE_USD
            minForcedRebalanceUSD = d[29],  // MIN_FORCED_REBALANCE_USD
            targetAdjustPercent  = d[0],    // TARGET_ADJUST_PERCENT
        )
    }
}

// ── Result ──────────────────────────────────────────────────────────────────

/**
 * Output of [evaluateRebalance].
 *
 * @property type         The action classification.
 * @property buyQty       Number of units to buy (0.0 when [type] is [RebalanceType.NONE]).
 * @property shortfallUSD Dollar gap between triggered-baseline and current value.
 * @property newBaseline  Updated baseline after adjustment, or the original
 *                        baseline if no action was taken.
 * @property updatedState New tracking state, or `null` to signal that the
 *                        caller should clear rebalance tracking for this asset.
 */
data class RebalanceResult(
    val type: RebalanceType,
    val buyQty: Double,
    val shortfallUSD: Double,
    val newBaseline: Double,
    val updatedState: RebalanceState?,
)

// ── Constants ───────────────────────────────────────────────────────────────

/** Hysteresis recovery buffer: requires 0.2% above trigger to clear. */
private const val RECOVERY_BUFFER = 1.002

/** Minimum deviation change to count as a positive or negative cycle move. */
private const val PRECISION_THRESHOLD = 0.0001

/** Fraction of available cash to use when cash-constrained (5% fee buffer). */
private const val CASH_CONSTRAINED_UTILIZATION = 0.95

/** Minimum asset value to consider rebalancing (zero-balance guard). */
private const val MIN_VALUE_USD = 1.0

// ── Pure evaluation function ────────────────────────────────────────────────

/**
 * Evaluate whether a rebalance action is needed for a single asset.
 *
 * This is a **pure function**: given the same inputs it always produces the
 * same [RebalanceResult].  No wall-clock time, no network, no randomness.
 *
 * @param state           Current tracking state, or `null` if not active.
 * @param totalValueUSD   Current USD value of the asset holding.
 * @param currentBaseline Current target baseline for this asset.
 * @param currentPrice    Current per-unit price of the asset.
 * @param nowMs           Current epoch time in milliseconds.
 * @param cashBalanceUSD  Available cash in the portfolio.
 * @param config          Genome-derived rebalance parameters.
 * @return [RebalanceResult] with the action, updated baseline, and new state.
 */
fun evaluateRebalance(
    state: RebalanceState?,
    totalValueUSD: Double,
    currentBaseline: Double,
    currentPrice: Double,
    nowMs: Long,
    cashBalanceUSD: Double,
    config: RebalanceConfig = RebalanceConfig(),
): RebalanceResult {
    // ── Guard: invalid baseline ─────────────────────────────────────────
    if (currentBaseline <= 0.0) {
        return RebalanceResult(RebalanceType.NONE, 0.0, 0.0, currentBaseline, null)
    }

    val triggerValue = currentBaseline * (1.0 - config.triggerPercent)

    // ── Zero-balance guard ──────────────────────────────────────────────
    if (totalValueUSD < MIN_VALUE_USD) {
        // Drop any active state -- asset was presumably fully sold.
        return RebalanceResult(RebalanceType.NONE, 0.0, 0.0, currentBaseline, null)
    }

    // ── Recovery check (with hysteresis) ────────────────────────────────
    val recoveryBuffer = if (state != null) RECOVERY_BUFFER else 1.0
    if (totalValueUSD >= triggerValue * recoveryBuffer) {
        // Value has recovered above the trigger threshold -- clear state.
        return RebalanceResult(RebalanceType.NONE, 0.0, 0.0, currentBaseline, null)
    }

    // ── Trigger entry ───────────────────────────────────────────────────
    val activeState = state ?: RebalanceState(
        triggered = true,
        triggeredAtMs = nowMs,
        posCycleCount = 0,
        attemptCount = 0,
        cooldownUntilMs = 0L,
        baselineWhenTriggered = currentBaseline,
        previousDeviation = (totalValueUSD - currentBaseline) / currentBaseline,
    )

    // ── Forced rebalance ────────────────────────────────────────────────
    val elapsed = nowMs - activeState.triggeredAtMs
    if (elapsed > config.forceTimeoutMs) {
        val shortfall = activeState.baselineWhenTriggered - totalValueUSD
        val buyUSD = shortfall * config.forceShortfallPercent
        if (buyUSD >= config.minForcedRebalanceUSD && currentPrice > 0.0) {
            val qty = buyUSD / currentPrice
            if (qty > 0.0 && cashBalanceUSD >= buyUSD) {
                val newBaseline = currentBaseline * (1.0 - config.targetAdjustPercent)
                // Forced rebalance clears state
                return RebalanceResult(
                    type = RebalanceType.FORCED,
                    buyQty = qty,
                    shortfallUSD = shortfall,
                    newBaseline = newBaseline,
                    updatedState = null,
                )
            }
        }
        // Forced conditions not met (insufficient funds or below minimum);
        // fall through to cycle tracking but keep state alive.
    }

    // ── Deviation cycle tracking ────────────────────────────────────────
    val currentDeviation = (totalValueUSD - currentBaseline) / currentBaseline
    val prevDev = activeState.previousDeviation
    val newCycleCount = if (prevDev != null) {
        when {
            currentDeviation > prevDev + PRECISION_THRESHOLD ->
                activeState.posCycleCount + 1
            currentDeviation < prevDev - PRECISION_THRESHOLD ->
                max(0, activeState.posCycleCount - 1)
            else -> activeState.posCycleCount
        }
    } else {
        activeState.posCycleCount
    }
    val stateAfterCycles = activeState.copy(
        posCycleCount = newCycleCount,
        previousDeviation = currentDeviation,
    )

    // ── Standard rebalance ──────────────────────────────────────────────
    if (newCycleCount >= config.positiveThreshold && nowMs >= activeState.cooldownUntilMs) {
        val shortfall = activeState.baselineWhenTriggered - totalValueUSD
        val partialRecovery = minOf(1.0, config.partialRecoveryPercent)
        var buyUSD = shortfall * partialRecovery

        // Cash-constrained clamping
        if (buyUSD > cashBalanceUSD) {
            if (cashBalanceUSD >= config.minPartialRebalanceUSD) {
                buyUSD = cashBalanceUSD * CASH_CONSTRAINED_UTILIZATION
            }
        }

        if (buyUSD >= config.minPartialRebalanceUSD && currentPrice > 0.0) {
            val qty = buyUSD / currentPrice
            if (qty > 0.0 && cashBalanceUSD >= buyUSD) {
                val newBaseline = currentBaseline * (1.0 - config.targetAdjustPercent)
                val newAttemptCount = activeState.attemptCount + 1

                // Decide whether to clear state or keep tracking
                val projectedValue = totalValueUSD + buyUSD
                val resolvedState: RebalanceState? = if (
                    projectedValue >= activeState.baselineWhenTriggered * (1.0 - config.triggerPercent)
                ) {
                    // Recovered enough -- clear state
                    null
                } else if (newAttemptCount >= config.maxAttempts) {
                    // Maxed out attempts -- enter cooldown, keep state alive
                    stateAfterCycles.copy(
                        attemptCount = newAttemptCount,
                        posCycleCount = 0,
                        previousDeviation = null,
                        cooldownUntilMs = nowMs + config.cooldownMs,
                    )
                } else {
                    // More attempts available -- reset cycle tracking
                    stateAfterCycles.copy(
                        attemptCount = newAttemptCount,
                        posCycleCount = 0,
                        previousDeviation = null,
                    )
                }

                return RebalanceResult(
                    type = RebalanceType.STANDARD,
                    buyQty = qty,
                    shortfallUSD = shortfall,
                    newBaseline = newBaseline,
                    updatedState = resolvedState,
                )
            }
        }
    }

    // ── No action this cycle -- return updated tracking state ───────────
    return RebalanceResult(
        type = RebalanceType.NONE,
        buyQty = 0.0,
        shortfallUSD = 0.0,
        newBaseline = currentBaseline,
        updatedState = stateAfterCycles,
    )
}
