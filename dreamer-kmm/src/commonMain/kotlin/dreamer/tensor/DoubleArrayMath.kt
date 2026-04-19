/**
 * Autovectorized numeric kernels for dreamer-kmm commonMain.
 *
 * Every function uses plain for-loops over [DoubleArray] so the
 * Kotlin/LLM (LLVM / WASM) backends can auto-vectorize the hot paths.
 * No JVM-specific APIs; only `kotlin.math` symbols available in commonMain.
 */

package dreamer.tensor

import kotlin.math.sqrt
import kotlin.math.pow

// ──────────────────────────────────────────────────────────────────────────────
// Central tendency & dispersion
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Arithmetic mean. Returns `NaN` for an empty array.
 */
fun mean(values: DoubleArray): Double {
    if (values.isEmpty()) return Double.NaN
    var sum = 0.0
    for (i in values.indices) {
        sum += values[i]
    }
    return sum / values.size
}

/**
 * Population variance. Returns `NaN` for an empty array.
 */
fun variance(values: DoubleArray): Double {
    if (values.isEmpty()) return Double.NaN
    if (values.size == 1) return 0.0
    val mu = mean(values)
    var ss = 0.0
    for (i in values.indices) {
        val d = values[i] - mu
        ss += d * d
    }
    return ss / values.size
}

/**
 * Population standard deviation.
 */
fun stdDev(values: DoubleArray): Double = sqrt(variance(values))

// ──────────────────────────────────────────────────────────────────────────────
// Exponential moving average
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Exponential Moving Average over [history] with the given [span].
 *
 * Uses the standard smoothing factor `α = 2 / (span + 1)`.
 * The first output element is seeded with [history][0].
 *
 * The loop is a plain running accumulator — friendly to SIMD unrolling
 * on the LLVM and WASM backends.
 */
fun ema(history: DoubleArray, span: Int): DoubleArray {
    require(span >= 1) { "span must be >= 1, got $span" }
    val n = history.size
    val out = DoubleArray(n)
    if (n == 0) return out

    val alpha = 2.0 / (span + 1)
    out[0] = history[0]
    for (i in 1 until n) {
        out[i] = alpha * history[i] + (1.0 - alpha) * out[i - 1]
    }
    return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Drag / decay kernels
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Symmetric SPAR drag — matches Dreamer 1.2.js lines 783-785 exactly.
 *
 * JS formula:
 * ```
 * gap = baseline - current
 * newBaseline = current + (gap * coefficient)
 * ```
 * This is *symmetric*: the drag applies whether current is above or below baseline.
 * The coefficient (e.g. 0.999968) controls how quickly the baseline converges
 * toward the current value.
 */
fun sparDrag(
    baseline: DoubleArray,
    current: DoubleArray,
    coefficient: Double
): DoubleArray {
    require(baseline.size == current.size) {
        "baseline.size=${baseline.size} != current.size=${current.size}"
    }
    val n = baseline.size
    val out = DoubleArray(n)
    for (i in 0 until n) {
        if (baseline[i] > 0.0 && current[i] > 0.0) {
            val gap = baseline[i] - current[i]
            out[i] = current[i] + (gap * coefficient)
        } else {
            out[i] = baseline[i]
        }
    }
    return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Portfolio metrics
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Root-mean-square deviation of [values] from corresponding [baselines].
 *
 * `sqrt( mean( (values[i] - baselines[i])^2 ) )`
 *
 * Both arrays must have the same length.
 */
fun portfolioDeviation(values: DoubleArray, baselines: DoubleArray): Double {
    require(values.size == baselines.size) {
        "values.size=${values.size} != baselines.size=${baselines.size}"
    }
    val n = values.size
    if (n == 0) return Double.NaN

    var ss = 0.0
    for (i in 0 until n) {
        val d = values[i] - baselines[i]
        ss += d * d
    }
    return sqrt(ss / n)
}

/**
 * Reinvestment score — amplifies or dampens deviations by an [exponent].
 *
 * `out[i] = sign(deviations[i]) * |deviations[i]|^exponent`
 *
 * Preserves sign while non-linearly scaling magnitude.
 */
fun reinvestScore(deviations: DoubleArray, exponent: Double): DoubleArray {
    val n = deviations.size
    val out = DoubleArray(n)
    for (i in 0 until n) {
        val v = deviations[i]
        out[i] = if (v >= 0.0) {
            v.pow(exponent)
        } else {
            -((-v).pow(exponent))
        }
    }
    return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Rolling window
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Sliding-window sum with width [window].
 *
 * Uses an O(n) prefix-sum (cumulative-sum) approach:
 *   `out[i] = prefix[i+1] - prefix[max(0, i+1-window)]`
 *
 * Elements that don't have a full window simply sum what is available
 * (partial window at the leading edge).
 */
fun windowSum(values: DoubleArray, window: Int): DoubleArray {
    require(window >= 1) { "window must be >= 1, got $window" }
    val n = values.size
    if (n == 0) return DoubleArray(0)

    // Build prefix sum: prefix[0] = 0, prefix[i+1] = sum of values[0..i]
    val prefix = DoubleArray(n + 1)
    for (i in 0 until n) {
        prefix[i + 1] = prefix[i] + values[i]
    }

    val out = DoubleArray(n)
    for (i in 0 until n) {
        val start = if (i + 1 >= window) i + 1 - window else 0
        out[i] = prefix[i + 1] - prefix[start]
    }
    return out
}
