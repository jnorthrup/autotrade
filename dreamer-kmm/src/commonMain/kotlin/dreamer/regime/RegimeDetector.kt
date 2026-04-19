/**
 * RegimeDetector — classifies market regime from a price history DoubleArray.
 *
 * Port of Dreamer 1.2.js RegimeDetector.analyze (lines 54-73) to Kotlin
 * Multiplatform commonMain using TrikeShed Series abstractions and
 * DoubleArrayMath.
 *
 * Pure functional: no side effects, no state, no allocation beyond the
 * result enum.
 */
package dreamer.regime

import borg.trikeshed.lib.Series
import borg.trikeshed.lib.size
import borg.trikeshed.lib.toSeries
import dreamer.tensor.mean
import dreamer.tensor.variance
import kotlin.math.sqrt

// ── Regime enum ──────────────────────────────────────────────────────────────

/**
 * Market regime states, matching the Dreamer kernel contract REGIME_CODES.
 */
enum class Regime(val code: Int) {
    /** Insufficient data or invalid input. */
    UNKNOWN(0),
    /** Sideways / range-bound with no strong trend. */
    CRAB_CHOP(1),
    /** Strong upward move with elevated volatility. */
    BULL_RUSH(2),
    /** Strong downward move with elevated volatility. */
    BEAR_CRASH(3),
    /** Gentle upward drift with low volatility. */
    STEADY_GROWTH(4),
    /** High volatility without clear directional trend. */
    VOLATILE_CHOP(5),
}

// ── Thresholds ───────────────────────────────────────────────────────────────

/** Encapsulates the five threshold constants used for regime classification. */
data class RegimeThresholds(
    val minHistory: Int = 50,
    val roiBull: Double = 0.05,
    val roiBear: Double = -0.05,
    val volDirectional: Double = 0.02,
    val roiSteady: Double = 0.02,
    val volSteady: Double = 0.01,
    val volChop: Double = 0.05,
)

// ── Intermediate statistics ─────────────────────────────────────────────────

/** Snapshot of the statistics driving regime classification. */
data class RegimeStats(
    val roi: Double,
    val volatility: Double,
    val meanPrice: Double,
)

// ── Detector ────────────────────────────────────────────────────────────────

/**
 * Pure-functional regime classifier.
 *
 * Usage:
 * ```
 *   val detector = RegimeDetector()
 *   val regime   = detector.analyze(priceHistory)
 * ```
 *
 * The [analyze] method converts the DoubleArray into a TrikeShed `Series<Double>`
 * via `.toSeries()`, then uses DoubleArrayMath for central-tendency and
 * dispersion, and applies the Dreamer threshold cascade.
 */
class RegimeDetector(
    private val thresholds: RegimeThresholds = RegimeThresholds(),
) {
    /**
     * Classify the market regime from a price [history].
     *
     * Returns [Regime.UNKNOWN] when the history is shorter than
     * [RegimeThresholds.minHistory] or when the mean price is non-positive
     * (which would make the coefficient of variation undefined).
     */
    fun analyze(history: DoubleArray): Regime =
        classify(history, thresholds)

    /**
     * Compute intermediate statistics for the history without classifying.
     *
     * Useful for diagnostics and testing. Returns `null` when the history
     * is too short or the mean is non-positive.
     */
    fun stats(history: DoubleArray): RegimeStats? {
        val t = thresholds
        if (history.size < t.minHistory) return null
        val series: Series<Double> = history.toSeries()

        // Use DoubleArrayMath for mean and variance
        val mu = mean(history)
        if (mu <= 0.0) return null
        val vol = sqrt(variance(history)) / mu

        // ROI: (last - first) / first
        val startPrice = history[0]
        val currentPrice = history[history.lastIndex]
        val roi = if (startPrice > 0.0) (currentPrice - startPrice) / startPrice else 0.0

        return RegimeStats(roi = roi, volatility = vol, meanPrice = mu)
    }
}

// ── Pure-functional classification ──────────────────────────────────────────

/**
 * Top-level pure function: classify a price history into a [Regime].
 *
 * Algorithm (mirrors Dreamer 1.2.js lines 54-73):
 *   1. Guard: needs >= [RegimeThresholds.minHistory] points.
 *   2. Compute mean and population variance over the full history via
 *      DoubleArrayMath.
 *   3. `volatility = sqrt(variance) / mean`  (coefficient of variation).
 *   4. `roi = (current - start) / start` where current = last element,
 *      start = first element.
 *   5. Threshold cascade:
 *      - roi > 0.05 && vol > 0.02  →  BULL_RUSH
 *      - roi < -0.05 && vol > 0.02 →  BEAR_CRASH
 *      - roi > 0.02 && vol < 0.01  →  STEADY_GROWTH
 *      - vol > 0.05                →  VOLATILE_CHOP
 *      - otherwise                 →  CRAB_CHOP
 */
fun classify(history: DoubleArray, t: RegimeThresholds = RegimeThresholds()): Regime {
    if (history.size < t.minHistory) return Regime.UNKNOWN

    // Convert to TrikeShed Series for index access
    val series: Series<Double> = history.toSeries()

    // Central tendency & dispersion via DoubleArrayMath
    val mu = mean(history)
    if (mu <= 0.0) return Regime.UNKNOWN
    val vol = sqrt(variance(history)) / mu

    // ROI from first to last element
    val startPrice = history[0]
    val currentPrice = history[history.lastIndex]
    val roi = if (startPrice > 0.0) (currentPrice - startPrice) / startPrice else 0.0

    // Threshold cascade (order matters — mirrors the JS precedence)
    return when {
        roi > t.roiBull && vol > t.volDirectional -> Regime.BULL_RUSH
        roi < t.roiBear && vol > t.volDirectional  -> Regime.BEAR_CRASH
        roi > t.roiSteady && vol < t.volSteady     -> Regime.STEADY_GROWTH
        vol > t.volChop                            -> Regime.VOLATILE_CHOP
        else                                       -> Regime.CRAB_CHOP
    }
}

// ── Series-based convenience ────────────────────────────────────────────────

/**
 * Extension on TrikeShed `Series<Double>` to compute regime statistics.
 *
 * Materialises the series into a DoubleArray for DoubleArrayMath, then
 * returns the intermediate stats. Uses the α (alpha-conversion) operator
 * to demonstrate the TrikeShed idiom for element-wise transforms.
 */
fun Series<Double>.regimeStats(thresholds: RegimeThresholds = RegimeThresholds()): RegimeStats? {
    if (size < thresholds.minHistory) return null

    // Materialise to DoubleArray for DoubleArrayMath kernels
    val arr = DoubleArray(size) { i -> b(i) }
    val mu = mean(arr)
    if (mu <= 0.0) return null
    val vol = sqrt(variance(arr)) / mu

    val startPrice = arr[0]
    val currentPrice = arr[arr.lastIndex]
    val roi = if (startPrice > 0.0) (currentPrice - startPrice) / startPrice else 0.0

    return RegimeStats(roi = roi, volatility = vol, meanPrice = mu)
}

/**
 * Classify regime directly from a TrikeShed `Series<Double>`.
 */
fun Series<Double>.classifyRegime(thresholds: RegimeThresholds = RegimeThresholds()): Regime =
    regimeStats(thresholds)?.let { stats -> classifyFromStats(stats, thresholds) }
        ?: Regime.UNKNOWN

/**
 * Pure threshold cascade over pre-computed stats.
 */
fun classifyFromStats(stats: RegimeStats, t: RegimeThresholds = RegimeThresholds()): Regime =
    when {
        stats.roi > t.roiBull && stats.volatility > t.volDirectional -> Regime.BULL_RUSH
        stats.roi < t.roiBear && stats.volatility > t.volDirectional  -> Regime.BEAR_CRASH
        stats.roi > t.roiSteady && stats.volatility < t.volSteady     -> Regime.STEADY_GROWTH
        stats.volatility > t.volChop                                  -> Regime.VOLATILE_CHOP
        else                                                          -> Regime.CRAB_CHOP
    }
