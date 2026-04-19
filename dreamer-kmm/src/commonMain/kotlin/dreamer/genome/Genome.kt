/**
 * Dreamer 1.2 evolutionary genome — flat DoubleArray-backed parameter pack.
 *
 * All tunable numeric parameters live in a single [DoubleArray] addressed by
 * ordinal constants in the companion.  This layout lets Kotlin/LLVM and WASM
 * backends auto-vectorize tight loops that sweep across parameters without
 * boxing overhead.
 *
 * Per-asset overrides are stored as a `Map<String, DoubleArray>` with the same
 * ordinal scheme; `Double.NaN` marks "use the global default".  Call
 * [resolve] or the zero-alloc [resolveInto] to produce a merged parameter
 * vector for a specific asset symbol.
 */

package dreamer.genome

import kotlin.math.abs
import kotlin.math.min

// ──────────────────────────────────────────────────────────────────────────────
// Genome data class
// ──────────────────────────────────────────────────────────────────────────────

data class Genome(
    /** Flat parameter vector addressed by companion ordinals. */
    val doubles: DoubleArray,

    // ── Boolean switches (control flow, not math) ──
    val allowMutation: Boolean = true,
    val enablePortfolioHarvest: Boolean = true,
    val enableCrashProtection: Boolean = true,
    val enableDeveloperLogs: Boolean = true,

    /**
     * Per-asset overrides.  Each value is a [DoubleArray] of the same
     * dimension as [doubles].  `Double.NaN` means "inherit global default".
     */
    val overrides: Map<String, DoubleArray> = emptyMap()
) {

    init {
        require(doubles.size == WIDTH) {
            "Genome doubles must have exactly $WIDTH elements, got ${doubles.size}"
        }
        overrides.forEach { (sym, arr) ->
            require(arr.size == WIDTH) {
                "Override for '$sym' must have $WIDTH elements, got ${arr.size}"
            }
        }
    }

    // ── Typed accessors ────────────────────────────────────────────────────

    /** Get parameter by ordinal. */
    fun d(ordinal: Int): Double = doubles[ordinal]

    /** Get parameter by ordinal with per-asset override resolution. */
    fun d(ordinal: Int, asset: String): Double {
        val ov = overrides[asset]
        if (ov != null) {
            val v = ov[ordinal]
            if (!v.isNaN()) return v
        }
        return doubles[ordinal]
    }

    // ── Per-asset resolution ───────────────────────────────────────────────

    /**
     * Merge global defaults with per-asset overrides into [out].
     * Zero-allocation hot-path: pass a reusable buffer.
     *
     * For each ordinal *i*:
     *   - if `overrides[asset][i]` is not NaN, use it;
     *   - otherwise fall back to `doubles[i]`.
     *
     * Returns [out] for chaining.
     */
    fun resolveInto(asset: String, out: DoubleArray): DoubleArray {
        val ov = overrides[asset]
        if (ov == null) {
            doubles.copyInto(out, 0, 0, WIDTH)
            return out
        }
        for (i in 0 until WIDTH) {
            val v = ov[i]
            out[i] = if (v.isNaN()) doubles[i] else v
        }
        return out
    }

    /**
     * Allocate a new merged parameter vector for [asset].
     * Convenience wrapper — prefer [resolveInto] in hot loops.
     */
    fun resolve(asset: String): DoubleArray =
        resolveInto(asset, DoubleArray(WIDTH))

    // ── Structural helpers ─────────────────────────────────────────────────

    /**
     * Create a copy with one ordinal changed.
     * Does NOT affect overrides — callers that need per-asset mutation
     * should use [withOverride].
     */
    fun withDouble(ordinal: Int, value: Double): Genome {
        val newDoubles = doubles.copyOf()
        newDoubles[ordinal] = value
        return copy(doubles = newDoubles)
    }

    /**
     * Set (or add) an override for a single ordinal on [asset].
     * Returns a new Genome; the original is untouched.
     */
    fun withOverride(asset: String, ordinal: Int, value: Double): Genome {
        val existing = overrides[asset]?.copyOf() ?: DoubleArray(WIDTH) { Double.NaN }
        existing[ordinal] = value
        val newOverrides = overrides + (asset to existing)
        return copy(overrides = newOverrides)
    }

    // ── equals / hashCode (deep for DoubleArray) ───────────────────────────

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Genome) return false
        if (!doubles.contentEquals(other.doubles)) return false
        if (allowMutation != other.allowMutation) return false
        if (enablePortfolioHarvest != other.enablePortfolioHarvest) return false
        if (enableCrashProtection != other.enableCrashProtection) return false
        if (enableDeveloperLogs != other.enableDeveloperLogs) return false
        if (overrides.size != other.overrides.size) return false
        for ((key, arr) in overrides) {
            val otherArr = other.overrides[key] ?: return false
            if (!arr.contentEquals(otherArr)) return false
        }
        return true
    }

    override fun hashCode(): Int {
        var result = doubles.contentHashCode()
        result = 31 * result + allowMutation.hashCode()
        result = 31 * result + enablePortfolioHarvest.hashCode()
        result = 31 * result + enableCrashProtection.hashCode()
        result = 31 * result + enableDeveloperLogs.hashCode()
        for ((key, arr) in overrides) {
            result = 31 * result + key.hashCode()
            result = 31 * result + arr.contentHashCode()
        }
        return result
    }

    // ── Validation ─────────────────────────────────────────────────────────

    /** Verify that harvest allocation percentages sum to ~1.0. */
    fun validateAllocSums(tolerance: Double = 0.001): Boolean {
        val sum = doubles[HARVEST_ALLOC_BTC_PERCENT] +
                doubles[HARVEST_ALLOC_ETH_PERCENT] +
                doubles[HARVEST_ALLOC_REINVEST_PERCENT] +
                doubles[HARVEST_ALLOC_CASH_PERCENT]
        return abs(sum - 1.0) <= tolerance
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Companion — ordinals & defaults
    // ──────────────────────────────────────────────────────────────────────────

    companion object {

        /** Number of packed double parameters. */
        const val WIDTH = 44

        // ── Core Strategy (0..0) ────────────────────────────────────────────
        const val TARGET_ADJUST_PERCENT = 0

        // ── Individual Asset Harvest (1..6) ─────────────────────────────────
        const val FLAT_HARVEST_TRIGGER_PERCENT = 1
        const val HARVEST_TAKE_PERCENT = 2
        const val HARVEST_CYCLE_THRESHOLD = 3
        const val MIN_SURPLUS_FOR_HARVEST = 4
        const val MIN_SURPLUS_FOR_FORCED_HARVEST = 5
        const val FORCED_HARVEST_TIMEOUT_MS = 6          // 45 min

        // ── Portfolio Override Harvest (7..9) ───────────────────────────────
        const val PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT = 7
        const val PORTFOLIO_HARVEST_CONFIRMATION_CYCLES = 8
        const val MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST = 9

        // ── Harvest Proceeds Allocation (10..13) ────────────────────────────
        const val HARVEST_ALLOC_BTC_PERCENT = 10
        const val HARVEST_ALLOC_ETH_PERCENT = 11
        const val HARVEST_ALLOC_REINVEST_PERCENT = 12
        const val HARVEST_ALLOC_CASH_PERCENT = 13

        // ── Crash Fund / Reinvest Thresholds (14..19) ──────────────────────
        const val CRASH_FUND_THRESHOLD_PERCENT = 14
        const val MIN_HARVEST_TO_ALLOCATE = 15
        const val MIN_NEGATIVE_DEVIATION_FOR_REINVEST = 16
        const val MIN_REINVEST_BUY_USD = 17
        const val REINVEST_BASELINE_GROWTH_FACTOR = 18
        const val MIN_BTC_BUY_USD = 19

        // ── Individual Asset Rebalance (20..21) ─────────────────────────────
        const val MIN_ETH_BUY_USD = 20
        const val FLAT_REBALANCE_TRIGGER_PERCENT = 21

        // ── Rebalance Mechanics (22..29) ────────────────────────────────────
        const val PARTIAL_RECOVERY_PERCENT = 22
        const val REBALANCE_POSITIVE_THRESHOLD = 23
        const val MAX_REBALANCE_ATTEMPTS = 24
        const val REBALANCE_COOLDOWN_MS = 25              // 30 min
        const val FORCE_REBALANCE_TIMEOUT_MS = 26          // 25 min
        const val FORCE_REBALANCE_SHORTFALL_PERCENT = 27
        const val MIN_PARTIAL_REBALANCE_USD = 28
        const val MIN_FORCED_REBALANCE_USD = 29

        // ── Project Dynamo / Physics (30) ───────────────────────────────────
        const val SPAR_DRAG_COEFFICIENT = 30

        // ── Crash Protection (31..34) ───────────────────────────────────────
        const val CP_TRIGGER_ASSET_PERCENT = 31
        const val CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT = 32
        const val CRASH_PROTECTION_THRESHOLD_INCREASE = 33
        const val CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT = 34

        // ── Timing (35) ─────────────────────────────────────────────────────
        const val REFRESH_INTERVAL_MS = 35

        // ── Smart Evolution Strategy (36..37) ───────────────────────────────
        const val ALLOCATION_MODE = 36
        const val REINVEST_WEIGHT_EXPONENT = 37

        // ── Evolution & Oracle (38..41) ─────────────────────────────────────
        const val FITNESS_DRAWDOWN_PENALTY = 38
        const val MIN_TRADES_FOR_PROMOTION = 39
        const val EVOLUTION_CONSISTENCY_COUNT = 40
        const val ORACLE_TREND_THRESHOLD = 41
        const val ORACLE_VOLATILITY_THRESHOLD = 42
        const val EVOLUTION_INTERVAL_MINUTES = 43

        // ── Named ordinal groups for batch operations ───────────────────────

        /** All harvest-allocation ordinals; sum should ≈ 1.0. */
        val HARVEST_ALLOC_ORDINALS = intArrayOf(
            HARVEST_ALLOC_BTC_PERCENT,
            HARVEST_ALLOC_ETH_PERCENT,
            HARVEST_ALLOC_REINVEST_PERCENT,
            HARVEST_ALLOC_CASH_PERCENT
        )

        /** Time-based ordinals stored in milliseconds. */
        val TIME_ORDINALS_MS = intArrayOf(
            FORCED_HARVEST_TIMEOUT_MS,
            REBALANCE_COOLDOWN_MS,
            FORCE_REBALANCE_TIMEOUT_MS,
            REFRESH_INTERVAL_MS
        )

        /** Integer-valued ordinals (safe to truncate with `.toInt()`). */
        val INTEGER_ORDINALS = intArrayOf(
            HARVEST_CYCLE_THRESHOLD,
            PORTFOLIO_HARVEST_CONFIRMATION_CYCLES,
            REBALANCE_POSITIVE_THRESHOLD,
            MAX_REBALANCE_ATTEMPTS,
            CRASH_PROTECTION_THRESHOLD_INCREASE,
            MIN_TRADES_FOR_PROMOTION,
            EVOLUTION_CONSISTENCY_COUNT,
            ALLOCATION_MODE,
            EVOLUTION_INTERVAL_MINUTES
        )

        // ── Default parameter vector (Dreamer 1.2.js defaultGenome) ─────────

        private const val MIN_45_MS = 45.0 * 60.0 * 1000.0
        private const val MIN_30_MS = 30.0 * 60.0 * 1000.0
        private const val MIN_25_MS = 25.0 * 60.0 * 1000.0

        val DEFAULT_DOUBLES = doubleArrayOf(
            // [0] Core Strategy
            0.001,          // TARGET_ADJUST_PERCENT
            // [1..6] Individual Asset Harvest
            0.035,          // FLAT_HARVEST_TRIGGER_PERCENT
            0.70,           // HARVEST_TAKE_PERCENT
            3.0,            // HARVEST_CYCLE_THRESHOLD
            0.25,           // MIN_SURPLUS_FOR_HARVEST
            1.00,           // MIN_SURPLUS_FOR_FORCED_HARVEST
            MIN_45_MS,      // FORCED_HARVEST_TIMEOUT_MS
            // [7..9] Portfolio Override Harvest
            0.035,          // PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT
            3.0,            // PORTFOLIO_HARVEST_CONFIRMATION_CYCLES
            0.10,           // MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST
            // [10..13] Harvest Proceeds Allocation
            0.25,           // HARVEST_ALLOC_BTC_PERCENT
            0.25,           // HARVEST_ALLOC_ETH_PERCENT
            0.25,           // HARVEST_ALLOC_REINVEST_PERCENT
            0.25,           // HARVEST_ALLOC_CASH_PERCENT
            // [14..19] Crash Fund / Reinvest
            0.10,           // CRASH_FUND_THRESHOLD_PERCENT
            0.25,           // MIN_HARVEST_TO_ALLOCATE
            -0.010,         // MIN_NEGATIVE_DEVIATION_FOR_REINVEST
            0.25,           // MIN_REINVEST_BUY_USD
            0.50,           // REINVEST_BASELINE_GROWTH_FACTOR
            0.10,           // MIN_BTC_BUY_USD
            // [20..21] Per-asset buy minimums / Rebalance trigger
            0.25,           // MIN_ETH_BUY_USD
            0.035,          // FLAT_REBALANCE_TRIGGER_PERCENT
            // [22..29] Rebalance Mechanics
            0.70,           // PARTIAL_RECOVERY_PERCENT
            3.0,            // REBALANCE_POSITIVE_THRESHOLD
            3.0,            // MAX_REBALANCE_ATTEMPTS
            MIN_30_MS,      // REBALANCE_COOLDOWN_MS
            MIN_25_MS,      // FORCE_REBALANCE_TIMEOUT_MS
            0.25,           // FORCE_REBALANCE_SHORTFALL_PERCENT
            0.25,           // MIN_PARTIAL_REBALANCE_USD
            0.25,           // MIN_FORCED_REBALANCE_USD
            // [30] Project Dynamo
            0.999968,       // SPAR_DRAG_COEFFICIENT
            // [31..34] Crash Protection
            0.70,           // CP_TRIGGER_ASSET_PERCENT
            -0.07,          // CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT
            2.0,            // CRASH_PROTECTION_THRESHOLD_INCREASE
            0.33,           // CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT
            // [35] Timing
            8000.0,         // REFRESH_INTERVAL_MS
            // [36..37] Smart Evolution Strategy
            1.0,            // ALLOCATION_MODE
            1.50,           // REINVEST_WEIGHT_EXPONENT
            // [38..43] Evolution & Oracle
            1.00,           // FITNESS_DRAWDOWN_PENALTY
            1.0,            // MIN_TRADES_FOR_PROMOTION
            3.0,            // EVOLUTION_CONSISTENCY_COUNT
            0.8,            // ORACLE_TREND_THRESHOLD
            2.0,            // ORACLE_VOLATILITY_THRESHOLD
            5.0             // EVOLUTION_INTERVAL_MINUTES
        )

        /** The canonical Dreamer 1.2 default genome. */
        val DEFAULT = Genome(DEFAULT_DOUBLES)

        // ── Vectorized batch helpers ────────────────────────────────────────

        /**
         * Copy [src] ordinals from [ordinals] range into [dst] starting at
         * [dstOffset].  Plain loop — friendly to SIMD auto-vectorization.
         */
        fun batchCopy(
            src: DoubleArray,
            dst: DoubleArray,
            ordinals: IntArray,
            dstOffset: Int = 0
        ) {
            for (i in ordinals.indices) {
                dst[dstOffset + i] = src[ordinals[i]]
            }
        }

        /**
         * Compute a weighted mutation step: `out[i] = base[i] + delta * noise[i]`.
         * Plain element-wise FMA loop for auto-vectorization.
         */
        fun mutateFma(
            base: DoubleArray,
            delta: Double,
            noise: DoubleArray,
            out: DoubleArray
        ) {
            for (i in 0 until min(base.size, min(noise.size, out.size))) {
                out[i] = base[i] + delta * noise[i]
            }
        }
    }
}
