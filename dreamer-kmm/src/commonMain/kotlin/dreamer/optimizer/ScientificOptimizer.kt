/**
 * ScientificOptimizer — pure-functional combinatorial parameter sweep.
 *
 * Port of Dreamer 1.2.js ScientificOptimizer (lines 2606–3250).
 * All I/O, filesystem, process.argv, and process.send calls have been
 * stripped; the caller is responsible for persistence and IPC.
 *
 * The optimizer operates in three sweep modes:
 *  1. GRID       — coarse H×R combinatorial search
 *  2. MICRO_GRID — high-precision crosshair verification around anchor
 *  3. FINE_TUNE  — sequential single-parameter sweep
 *
 * @see dreamer.engine.runCycle
 * @see dreamer.genome.Genome
 */
package dreamer.optimizer

import dreamer.engine.CycleResult
import dreamer.engine.EngineMode
import dreamer.engine.EngineState
import dreamer.engine.PortfolioInput
import dreamer.engine.runCycle
import dreamer.genome.Genome
import dreamer.risk.RiskState
import dreamer.signal.PortfolioHarvestState
import kotlin.math.max
import kotlin.math.min

// ── Enums ──────────────────────────────────────────────────────────────────

/** Sweep mode for the combinatorial optimizer. */
enum class SweepMode {
    GRID,
    MICRO_GRID,
    FINE_TUNE,
}

/** History depth for simulation evaluation. */
enum class DepthMode {
    SHORT,
    MEDIUM,
    LONG,
}

// ── Data classes ───────────────────────────────────────────────────────────

/**
 * A numeric range for grid/fine-tune sweeps.
 *
 * Default ranges match Dreamer 1.2.js GRID_RANGES and FINE_TUNE_RANGES:
 *   - H: (0.02, 0.08, 0.005)
 *   - R: (0.02, 0.08, 0.005)
 *   - PARTIAL_RECOVERY: (0.30, 1.00, 0.10)
 */
data class GridRange(
    val start: Double,
    val end: Double,
    val step: Double,
) {
    /** Number of discrete steps (inclusive of end). */
    val stepCount: Int get() = ((end - start) / step).toInt()
}

/** Default grid ranges matching Dreamer 1.2.js. */
val DEFAULT_HARVEST_GRID_RANGE = GridRange(0.02, 0.08, 0.005)
val DEFAULT_REBALANCE_GRID_RANGE = GridRange(0.02, 0.08, 0.005)
val DEFAULT_PARTIAL_RECOVERY_RANGE = GridRange(0.30, 1.00, 0.10)

/**
 * Immutable state for the combinatorial sweep cursor.
 *
 * Port of JS SweepStateManager.state.
 */
data class SweepState(
    val mode: SweepMode = SweepMode.GRID,
    val hIndex: Int = 0,
    val rIndex: Int = 0,
    val currentAssetIndex: Int = 0,
    val combinationsChecked: Int = 0,
    val championChecked: Boolean = false,
    val anchorH: Double = 0.035,
    val anchorR: Double = 0.035,
    val microPhase: String? = null,  // "SWEEP_H" or "SWEEP_R"
    val mIndex: Int = 0,
    val paramIndex: Int = 0,
    val `val`: Double? = null,
)

/**
 * A candidate genome + metadata produced by [getNextCandidate].
 */
data class CandidateResult(
    val genome: Genome,
    val focus: String,
    val mode: SweepMode,
    val desc: String = "",
    val valStr: String = "",
)

/**
 * Result of simulating a genome over a tick history window.
 */
data class SimulationResult(
    val roi: Double,
    val relativeROI: Double,
    val marketROI: Double,
    val drawdown: Double,
    val totalTrades: Int,
    val totalValue: Double,
)

// ── Constants ──────────────────────────────────────────────────────────────

/** Maximum allowed rebalance trigger (safety cap). JS line 2827 / 2909. */
private const val MAX_REBALANCE_TRIGGER = 0.15

/** Micro-grid crosshair half-window. JS line 2873. */
private const val MICRO_RANGE = 0.005

/** Micro-grid step precision. JS line 2874. */
private const val MICRO_STEP = 0.0001

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Compute the next candidate genome to evaluate in the combinatorial sweep.
 *
 * Pure-functional: takes the current [state] and returns a pair of
 * `CandidateResult?` (null when no assets or invalid state) and the
 * updated [SweepState].
 *
 * Algorithm (mirrors JS lines 2779–3008):
 *  1. If GRID mode and hIndex=0, rIndex=0, champion not yet checked:
 *     → return base genome as-is ("champion check"), mark championChecked.
 *  2. GRID mode: compute H/R from index, apply overrides, advance counters.
 *     When grid is exhausted → transition to MICRO_GRID with anchors.
 *  3. MICRO_GRID mode: crosshair sweep (SWEEP_H then SWEEP_R) around anchors.
 *     When both phases complete → transition to FINE_TUNE.
 *  4. FINE_TUNE mode: sequential parameter sweep.
 *     When all params exhausted → advance asset, reset to GRID.
 *
 * @param state         Current sweep state.
 * @param assets        Sorted list of asset symbols to sweep.
 * @param baseGenome    The base genome to modify (typically live champion).
 * @param gridRanges    Pair of (harvestGrid, rebalanceGrid).
 * @param fineTuneRanges Map of parameter name → GridRange for fine-tuning.
 * @return Pair of (candidate or null, updated state).
 */
fun getNextCandidate(
    state: SweepState,
    assets: List<String>,
    baseGenome: Genome,
    gridRanges: Pair<GridRange, GridRange> = Pair(DEFAULT_HARVEST_GRID_RANGE, DEFAULT_REBALANCE_GRID_RANGE),
    fineTuneRanges: Map<String, GridRange> = mapOf("PARTIAL_RECOVERY_PERCENT" to DEFAULT_PARTIAL_RECOVERY_RANGE),
): Pair<CandidateResult?, SweepState> {
    if (assets.isEmpty()) return Pair(null, state)

    var st = state.copy(
        combinationsChecked = state.combinationsChecked + 1,
    )
    val asset = assets[st.currentAssetIndex % assets.size]

    // ── Champion Check (JS lines 2803–2814) ──────────────────────────────
    if (st.mode == SweepMode.GRID && st.hIndex == 0 && st.rIndex == 0 && !st.championChecked) {
        st = st.copy(championChecked = true)
        return Pair(
            CandidateResult(
                genome = baseGenome,
                focus = asset,
                mode = SweepMode.GRID,
                desc = "CHAMPION",
                valStr = "INCUMBENT",
            ),
            st,
        )
    }

    // ── GRID mode (JS lines 2817–2865) ───────────────────────────────────
    if (st.mode == SweepMode.GRID) {
        val hRange = gridRanges.first
        val rRange = gridRanges.second

        val hVal = hRange.start + (st.hIndex * hRange.step)
        val rVal = rRange.start + (st.rIndex * rRange.step)
        val constrainedRVal = min(rVal, MAX_REBALANCE_TRIGGER)

        val candidateGenome = baseGenome
            .withOverride(asset, Genome.FLAT_HARVEST_TRIGGER_PERCENT, roundTo4(hVal))
            .withOverride(asset, Genome.FLAT_REBALANCE_TRIGGER_PERCENT, roundTo4(constrainedRVal))

        val desc = "H:${formatPct(hVal)}% / R:${formatPct(constrainedRVal)}%"

        // Advance counters
        if (rVal >= rRange.end) {
            if (hVal >= hRange.end) {
                // Grid Complete → MICRO_GRID
                st = st.copy(
                    mode = SweepMode.MICRO_GRID,
                    hIndex = 0,
                    rIndex = 0,
                    anchorH = 0.035,
                    anchorR = 0.035,
                    microPhase = "SWEEP_H",
                    mIndex = 0,
                )
            } else {
                st = st.copy(rIndex = 0, hIndex = st.hIndex + 1)
            }
        } else {
            st = st.copy(rIndex = st.rIndex + 1)
        }

        return Pair(
            CandidateResult(
                genome = candidateGenome,
                focus = asset,
                mode = SweepMode.GRID,
                desc = desc,
                valStr = "COMBINED",
            ),
            st,
        )
    }

    // ── MICRO_GRID mode (JS lines 2869–2928) ─────────────────────────────
    if (st.mode == SweepMode.MICRO_GRID) {
        val anchorH = st.anchorH
        val anchorR = st.anchorR
        val steps = ((MICRO_RANGE * 2.0) / MICRO_STEP).toInt()

        var microPhase = st.microPhase ?: "SWEEP_H"
        var mIndex = st.mIndex

        var hVal: Double
        var rVal: Double
        var desc: String
        var valStr: String
        var candidateGenome: Genome

        if (microPhase == "SWEEP_H") {
            val start = max(0.01, anchorH - MICRO_RANGE)
            hVal = start + (mIndex * MICRO_STEP)
            rVal = anchorR

            candidateGenome = baseGenome
                .withOverride(asset, Genome.FLAT_HARVEST_TRIGGER_PERCENT, roundTo4(hVal))
                .withOverride(asset, Genome.FLAT_REBALANCE_TRIGGER_PERCENT, roundTo4(rVal))
            desc = "µH:${formatPct(hVal)}% (R Fixed)"
            valStr = "MICRO_H"

            mIndex++
            if (mIndex > steps) {
                microPhase = "SWEEP_R"
                mIndex = 0
            }

            st = st.copy(microPhase = microPhase, mIndex = mIndex)
        } else {
            // SWEEP_R
            val start = max(0.01, anchorR - MICRO_RANGE)
            hVal = anchorH
            rVal = min(start + (mIndex * MICRO_STEP), MAX_REBALANCE_TRIGGER)

            candidateGenome = baseGenome
                .withOverride(asset, Genome.FLAT_HARVEST_TRIGGER_PERCENT, roundTo4(hVal))
                .withOverride(asset, Genome.FLAT_REBALANCE_TRIGGER_PERCENT, roundTo4(rVal))
            desc = "µR:${formatPct(rVal)}% (H Fixed)"
            valStr = "MICRO_R"

            mIndex++
            if (mIndex > steps) {
                // Micro Grid Complete → FINE_TUNE
                st = st.copy(
                    mode = SweepMode.FINE_TUNE,
                    microPhase = null,
                    mIndex = 0,
                    paramIndex = 0,
                    `val` = null,
                )
            } else {
                st = st.copy(mIndex = mIndex)
            }
        }

        return Pair(
            CandidateResult(
                genome = candidateGenome,
                focus = asset,
                mode = SweepMode.MICRO_GRID,
                desc = desc,
                valStr = valStr,
            ),
            st,
        )
    }

    // ── FINE_TUNE mode (JS lines 2931–3001) ──────────────────────────────
    val keys = fineTuneRanges.keys.toList()
    if (keys.isEmpty()) {
        // No fine-tune parameters → advance asset, reset to GRID
        st = st.copy(
            mode = SweepMode.GRID,
            paramIndex = 0,
            `val` = null,
            currentAssetIndex = st.currentAssetIndex + 1,
            championChecked = false,
            combinationsChecked = 0,
        )
        return Pair(null, st)
    }

    if (st.paramIndex < 0 || st.paramIndex >= keys.size) {
        st = st.copy(paramIndex = 0, `val` = null)
    }

    val key = keys[st.paramIndex % keys.size]
    val range = fineTuneRanges[key]

    if (range == null) {
        // Corrupted → fallback to GRID
        st = st.copy(mode = SweepMode.GRID, paramIndex = 0, `val` = null)
        return Pair(null, st)
    }

    // Resolve Genome ordinal from parameter key name
    val ordinal = resolveOrdinal(key)
    if (ordinal < 0) {
        st = st.copy(paramIndex = st.paramIndex + 1, `val` = null)
        return Pair(null, st)
    }

    var value = st.`val`
    if (value == null) {
        value = range.start
    } else {
        value += range.step
    }

    val candidateGenome = baseGenome.withOverride(asset, ordinal, roundTo4(value))
    val desc = key
    val valStr = roundTo2(value).toString()

    // Advance counters
    if (value >= range.end - 0.0001) {
        val nextParamIndex = st.paramIndex + 1
        if (nextParamIndex >= keys.size) {
            // Asset Complete → Next Asset → Back to GRID
            st = st.copy(
                paramIndex = 0,
                `val` = null,
                currentAssetIndex = st.currentAssetIndex + 1,
                mode = SweepMode.GRID,
                championChecked = false,
                combinationsChecked = 0,
            )
        } else {
            st = st.copy(paramIndex = nextParamIndex, `val` = null)
        }
    } else {
        st = st.copy(`val` = value)
    }

    return Pair(
        CandidateResult(
            genome = candidateGenome,
            focus = asset,
            mode = SweepMode.FINE_TUNE,
            desc = desc,
            valStr = valStr,
        ),
        st,
    )
}

/**
 * Simulate a genome over historical tick data and compute performance metrics.
 *
 * Port of JS evaluateGenome (lines 3159–3249).
 *
 * Algorithm:
 *  1. Select a history slice based on [depthMode]:
 *     - SHORT = 15k ticks (~24h)
 *     - MEDIUM = 45k ticks (~3d)
 *     - LONG = 105k ticks (~7d)
 *  2. If [holdingsQty] > 0 and [liveBaseline] > 0, reconstruct the "pain"
 *     by applying the live deviation ratio to the simulation start baseline.
 *     Otherwise, simulate a fresh $10k all-in on the focus asset.
 *  3. Run tick-by-tick simulation via [runCycle].
 *  4. Compute ROI, marketROI, relativeROI = roi − marketROI.
 *  5. Return [SimulationResult] with drawdown from risk metrics.
 *
 * @param genome        Genome to evaluate (may contain per-asset overrides).
 * @param ticks         Historical price data; each element is a Map of
 *                      symbol → price (equivalent to JS `tick.p`).
 * @param focusAsset    The single asset to simulate.
 * @param depthMode     SHORT, MEDIUM, or LONG history window.
 * @param initialCash   Starting cash (default 10000.0).
 * @param holdingsQty   Current holdings quantity for reality injection, or 0.0.
 * @param liveBaseline  Live baseline value for the focus asset, or 0.0.
 * @param livePrice     Live price at the time of snapshot, or 0.0.
 * @return SimulationResult with performance metrics.
 */
fun evaluateGenome(
    genome: Genome,
    ticks: List<Map<String, Double>>,
    focusAsset: String,
    depthMode: DepthMode,
    initialCash: Double = 10000.0,
    holdingsQty: Double = 0.0,
    liveBaseline: Double = 0.0,
    livePrice: Double = 0.0,
): SimulationResult {
    // ── Step 1: Select history slice (JS lines 3166–3168) ───────────────
    val windowSize = when (depthMode) {
        DepthMode.SHORT -> 15_000
        DepthMode.MEDIUM -> 45_000
        DepthMode.LONG -> 105_000
    }
    val sliceStart = max(0, ticks.size - windowSize)
    val historySlice = ticks.subList(sliceStart, ticks.size)

    // Safety: ensure we have data and a valid start price (JS lines 3171–3173)
    if (historySlice.isEmpty()) {
        return SimulationResult(
            roi = 0.0,
            relativeROI = 0.0,
            marketROI = 0.0,
            drawdown = 0.0,
            totalTrades = 0,
            totalValue = 0.0,
        )
    }

    val simStartPrice = historySlice[0][focusAsset]
    if (simStartPrice == null || simStartPrice <= 0.0) {
        return SimulationResult(
            roi = 0.0,
            relativeROI = 0.0,
            marketROI = 0.0,
            drawdown = 0.0,
            totalTrades = 0,
            totalValue = 0.0,
        )
    }

    // ── Step 2: Initialize engine state (JS lines 3175–3225) ────────────

    var startCapital = initialCash
    var initQty: Double
    var initBaseline: Double

    if (holdingsQty > 0.0 && liveBaseline > 0.0) {
        // Reality injection (JS lines 3181–3218)
        initQty = holdingsQty

        // Set start capital proportionally (JS line 3196)
        // The JS divides cash by total assets; we accept it as a parameter directly.

        // Reconstruct the "pain" — deviation ratio (JS lines 3203–3211)
        val effectiveLivePrice = if (livePrice > 0.0) livePrice else simStartPrice
        val deviationRatio = if (liveBaseline > 0.0) {
            liveBaseline / (holdingsQty * effectiveLivePrice)
        } else {
            1.0
        }

        initBaseline = (holdingsQty * simStartPrice) * deviationRatio
    } else {
        // Fresh $10k all-in (JS lines 3220–3224)
        initQty = startCapital / simStartPrice
        initBaseline = startCapital
    }

    // Build initial engine state
    val symbols = arrayOf(focusAsset)
    val initPrices = doubleArrayOf(simStartPrice)
    val initValues = doubleArrayOf(initQty * simStartPrice)
    val initQuantities = doubleArrayOf(initQty)
    val initBaselines = doubleArrayOf(initBaseline)

    var engineState = EngineState(
        baselines = initBaselines,
        trailingStates = emptyMap(),
        rebalanceStates = emptyMap(),
        portfolioHarvestState = PortfolioHarvestState(),
        lastActionTimestamps = emptyMap(),
        riskState = RiskState(),
        cashBalance = startCapital,
        lastTotalValue = initQty * simStartPrice + startCapital,
        lastCyclePrices = initPrices.copyOf(),
        priceHistory = emptyMap(),
        totalHarvested = 0.0,
        totalTrades = 0,
    )

    // ── Step 3: Run simulation (JS lines 3230–3235) ─────────────────────

    var cumulativeTrades = 0

    for (tick in historySlice) {
        val price = tick[focusAsset]
        if (price == null || price <= 0.0) continue

        val qty = if (engineState.baselines.isNotEmpty()) {
            // Track quantity through trades: initial qty adjusted by trades
            // We need to maintain qty ourselves since EngineState doesn't hold it
            initQty
        } else {
            0.0
        }

        val value = qty * price

        val portfolioInput = PortfolioInput(
            symbols = symbols,
            prices = doubleArrayOf(price),
            values = doubleArrayOf(value),
            quantities = doubleArrayOf(qty),
        )

        val cycleResult: CycleResult = runCycle(
            input = portfolioInput,
            state = engineState,
            genome = genome,
            nowMs = 0L,
            mode = EngineMode.SHADOW,
        )

        engineState = cycleResult.updatedState
        cumulativeTrades = engineState.totalTrades

        // Update tracked quantity based on trades
        for (trade in cycleResult.trades) {
            if (trade.asset == focusAsset) {
                when (trade.side) {
                    "BUY" -> initQty += trade.quantity
                    "SELL" -> initQty -= trade.quantity
                }
            }
        }
    }

    // ── Step 4: Compute metrics (JS lines 3237–3248) ────────────────────

    val finalPrice = historySlice.last()[focusAsset] ?: simStartPrice
    val finalVal = engineState.cashBalance + (initQty * finalPrice)

    // Initial total equity = startCash + (initial holdings qty × start price)
    val initialTotalEquity = initialCash + (holdingsQty.let { if (it > 0.0) it else (initialCash / simStartPrice) } * simStartPrice)
    val roi = if (initialTotalEquity > 0.0) {
        ((finalVal - initialTotalEquity) / initialTotalEquity) * 100.0
    } else {
        0.0
    }

    val marketROI = if (simStartPrice > 0.0) {
        ((finalPrice - simStartPrice) / simStartPrice) * 100.0
    } else {
        0.0
    }

    val relativeROI = roi - marketROI

    return SimulationResult(
        roi = roi,
        relativeROI = relativeROI,
        marketROI = marketROI,
        drawdown = engineState.riskState.maxDrawdownPercent,
        totalTrades = cumulativeTrades,
        totalValue = finalVal,
    )
}

/**
 * Score a simulation result for promotion decisions.
 *
 * Port of JS scoring logic (lines 2718–2727):
 *  - If totalTrades < MIN_TRADES_FOR_PROMOTION → -1000.0
 *  - penalty = 1 + drawdown * FITNESS_DRAWDOWN_PENALTY
 *  - score = relativeROI / penalty
 *  - Only positive scores (> 0.0001) are valid for promotion
 *
 * @param result  The simulation result to score.
 * @param genome  The genome (for MIN_TRADES_FOR_PROMOTION and FITNESS_DRAWDOWN_PENALTY).
 * @return Fitness score; positive values indicate valid promotion candidates.
 */
fun scoreCandidate(result: SimulationResult, genome: Genome): Double {
    if (result.totalValue <= -1.0) return Double.NEGATIVE_INFINITY

    val minTrades = genome.d(Genome.MIN_TRADES_FOR_PROMOTION).let { v ->
        if (v.isNaN() || v == 0.0) 1.0 else v
    }

    if (result.totalTrades < minTrades.toInt()) return -1000.0

    val dd = max(0.01, result.drawdown)
    val penaltyCoeff = genome.d(Genome.FITNESS_DRAWDOWN_PENALTY).let { v ->
        if (v.isNaN()) 1.0 else v
    }
    val penalty = 1.0 + dd * penaltyCoeff
    val score = result.relativeROI / penalty

    return score
}

// ── Internal helpers ───────────────────────────────────────────────────────

/** Round a double to 4 decimal places (mirrors JS toFixed(4)). */
private fun roundTo4(v: Double): Double {
    val factor = 10_000.0
    return kotlin.math.round(v * factor) / factor
}

/** Round a double to 2 decimal places. */
private fun roundTo2(v: Double): Double {
    val factor = 100.0
    return kotlin.math.round(v * factor) / factor
}

/** Format a proportion as a percentage string with 2 decimal places. commonMain-safe. */
private fun formatPct(v: Double): String {
    val pct = v * 100.0
    val intPart = pct.toLong()
    var frac = ((pct - intPart.toDouble()) * 100.0).toLong().let { if (it < 0) -it else it }
    val fracStr = frac.toString().padStart(2, '0')
    return "$intPart.$fracStr"
}

/**
 * Resolve a fine-tune parameter key name to a Genome ordinal.
 * Returns -1 if unknown.
 */
private fun resolveOrdinal(key: String): Int = when (key) {
    "HARVEST_CYCLE_THRESHOLD" -> Genome.HARVEST_CYCLE_THRESHOLD
    "REBALANCE_POSITIVE_THRESHOLD" -> Genome.REBALANCE_POSITIVE_THRESHOLD
    "PARTIAL_RECOVERY_PERCENT" -> Genome.PARTIAL_RECOVERY_PERCENT
    "FLAT_HARVEST_TRIGGER_PERCENT" -> Genome.FLAT_HARVEST_TRIGGER_PERCENT
    "FLAT_REBALANCE_TRIGGER_PERCENT" -> Genome.FLAT_REBALANCE_TRIGGER_PERCENT
    "HARVEST_TAKE_PERCENT" -> Genome.HARVEST_TAKE_PERCENT
    "MIN_SURPLUS_FOR_HARVEST" -> Genome.MIN_SURPLUS_FOR_HARVEST
    "MIN_SURPLUS_FOR_FORCED_HARVEST" -> Genome.MIN_SURPLUS_FOR_FORCED_HARVEST
    "FORCE_REBALANCE_SHORTFALL_PERCENT" -> Genome.FORCE_REBALANCE_SHORTFALL_PERCENT
    "CP_TRIGGER_ASSET_PERCENT" -> Genome.CP_TRIGGER_ASSET_PERCENT
    "CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT" -> Genome.CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT
    "CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT" -> Genome.CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT
    "CRASH_FUND_THRESHOLD_PERCENT" -> Genome.CRASH_FUND_THRESHOLD_PERCENT
    "MIN_NEGATIVE_DEVIATION_FOR_REINVEST" -> Genome.MIN_NEGATIVE_DEVIATION_FOR_REINVEST
    "REINVEST_BASELINE_GROWTH_FACTOR" -> Genome.REINVEST_BASELINE_GROWTH_FACTOR
    "TARGET_ADJUST_PERCENT" -> Genome.TARGET_ADJUST_PERCENT
    "SPAR_DRAG_COEFFICIENT" -> Genome.SPAR_DRAG_COEFFICIENT
    "REINVEST_WEIGHT_EXPONENT" -> Genome.REINVEST_WEIGHT_EXPONENT
    else -> -1
}
