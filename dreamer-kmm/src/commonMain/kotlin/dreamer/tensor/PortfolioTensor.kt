/**
 * PortfolioTensor — autovectorized portfolio state packed into a dense row-major tensor.
 *
 * Wraps a [WasmDoubleTensor]-style flat [DoubleArray] where:
 *   - rows   = assets (one per tracked symbol)
 *   - columns = 5 fields: [PRICE, VALUE, BASELINE, QUANTITY, DEVIATION]
 *
 * A companion [Array]<String> maps row index to symbol for O(1) lookup.
 * Row access is exposed as a TrikeShed [Series]<[AssetRow]> for lazy iteration
 * without intermediate allocations.
 *
 * All loops are plain `for (i in …)` so the Kotlin/LLVM and WASM backends
 * can auto-vectorize the hot paths.  No JVM-specific APIs.
 *
 * @see DoubleArrayMath for the underlying numeric kernels.
 */

package dreamer.tensor

import borg.trikeshed.lib.Join
import borg.trikeshed.lib.Series
import borg.trikeshed.lib.j
import borg.trikeshed.lib.size

// ──────────────────────────────────────────────────────────────────────────────
// Column ordinals
// ──────────────────────────────────────────────────────────────────────────────

/** Number of packed fields per asset row. */
private const val COLS = 5

/** Column ordinals into the flat [DoubleArray] row-major layout. */
object PortfolioCol {
    const val PRICE     = 0
    const val VALUE     = 1
    const val BASELINE  = 2
    const val QUANTITY  = 3
    const val DEVIATION = 4
}

// ──────────────────────────────────────────────────────────────────────────────
// Row view — lazy, zero-copy
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Light-weight view over a single asset row in the backing [DoubleArray].
 *
 * Does **not** copy data; reads from `data[offset + col]` on every access.
 * Instances are created per-access by the [Series]<[AssetRow]> lambda —
 * short-lived and allocation-friendly on all KMM backends.
 */
class AssetRow(
    private val data: DoubleArray,
    private val offset: Int,
) {
    val price:     Double get() = data[offset + PortfolioCol.PRICE]
    val value:     Double get() = data[offset + PortfolioCol.VALUE]
    val baseline:  Double get() = data[offset + PortfolioCol.BASELINE]
    val quantity:  Double get() = data[offset + PortfolioCol.QUANTITY]
    val deviation: Double get() = data[offset + PortfolioCol.DEVIATION]

    /** Write a single column value into this row. */
    fun set(col: Int, v: Double) {
        data[offset + col] = v
    }

    override fun toString(): String =
        "AssetRow(price=$price, value=$value, baseline=$baseline, " +
            "qty=$quantity, dev=$deviation)"
}

// ──────────────────────────────────────────────────────────────────────────────
// PortfolioTensor
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Immutable snapshot of portfolio state packed into a row-major [DoubleArray].
 *
 * Layout: `data[ row * COLS + col ]` where `row ∈ [0, symbols.size)` and
 * `col ∈ {PRICE, VALUE, BASELINE, QUANTITY, DEVIATION}`.
 *
 * Every query method returns a **new** array or a new [PortfolioTensor];
 * the backing data is never mutated in place.  This keeps the class safe
 * for concurrent reads across KMM targets.
 *
 * @param symbols  Asset symbol strings; index corresponds to tensor row.
 * @param data     Row-major packed DoubleArray of length `symbols.size * COLS`.
 */
class PortfolioTensor(
    val symbols: Array<String>,
    val data: DoubleArray,
) {

    /** Number of assets (rows) in the tensor. */
    val assetCount: Int get() = symbols.size

    init {
        require(data.size == symbols.size * COLS) {
            "data.size ${data.size} != symbols.size ${symbols.size} * $COLS"
        }
    }

    // ── Row-major indexing helpers ─────────────────────────────────────────

    /** Flat index for (row, col). Inline for autovectorization friendliness. */
    private inline fun idx(row: Int, col: Int): Int = row * COLS + col

    /** Read a single cell. */
    operator fun get(row: Int, col: Int): Double = data[idx(row, col)]

    /** Write a single cell into a **copy** of the data and return a new tensor. */
    private fun withCell(row: Int, col: Int, value: Double): PortfolioTensor {
        val copy = data.copyOf()
        copy[idx(row, col)] = value
        return PortfolioTensor(symbols, copy)
    }

    // ── Series<AssetRow> — lazy row access ─────────────────────────────────

    /**
     * TrikeShed [Series]<[AssetRow]> providing lazy, zero-copy row views.
     *
     * Usage:
     * ```
     *   val rows: Series<AssetRow> = tensor.rows
     *   for (i in 0 until rows.size) {
     *       val row = rows[i]
     *       println("symbol=${tensor.symbols[i]}  deviation=${row.deviation}")
     *   }
     * ```
     */
    val rows: Series<AssetRow> get() = assetCount j { row: Int ->
        AssetRow(data, row * COLS)
    }

    // ── Column extraction ──────────────────────────────────────────────────

    /**
     * Extract a full column as a [DoubleArray].
     *
     * Plain strided loop — friendly to SIMD auto-vectorization.
     *
     * @param col  Column ordinal from [PortfolioCol].
     * @return [DoubleArray] of length [assetCount].
     */
    fun column(col: Int): DoubleArray {
        val n = assetCount
        val out = DoubleArray(n)
        for (i in 0 until n) {
            out[i] = data[idx(i, col)]
        }
        return out
    }

    // ── deviation(col): DoubleArray ────────────────────────────────────────

    /**
     * Compute per-asset deviation ratios for a given column pair:
     * `deviation[i] = (values[i] - baselines[i]) / baselines[i]`
     *
     * The "col" parameter selects which column plays the role of "value"
     * while [PortfolioCol.BASELINE] is always the denominator.
     *
     * @param col  Column to use as the numerator (typically [PortfolioCol.VALUE]).
     * @return Per-asset deviation ratios. `Double.NaN` where baseline ≤ 0.
     */
    fun deviation(col: Int = PortfolioCol.VALUE): DoubleArray {
        val n = assetCount
        val out = DoubleArray(n)
        for (i in 0 until n) {
            val baseline = data[idx(i, PortfolioCol.BASELINE)]
            val value    = data[idx(i, col)]
            out[i] = if (baseline > 0.0) (value - baseline) / baseline else Double.NaN
        }
        return out
    }

    // ── harvestCandidates(): IntArray ──────────────────────────────────────

    /**
     * Identify asset indices eligible for harvest.
     *
     * An asset is a harvest candidate when its value exceeds
     * `baseline * (1 + trigger)`.
     *
     * @param trigger  Trigger fraction (e.g. 0.035 for 3.5 %). Defaults to
     *                 the Dreamer 1.2 genome default
     *                 [Genome.FLAT_HARVEST_TRIGGER_PERCENT] = 0.035.
     * @return Sorted [IntArray] of candidate row indices.
     */
    fun harvestCandidates(trigger: Double = 0.035): IntArray {
        val n = assetCount
        // First pass: count
        var count = 0
        for (i in 0 until n) {
            val baseline = data[idx(i, PortfolioCol.BASELINE)]
            val value    = data[idx(i, PortfolioCol.VALUE)]
            if (baseline > 0.0 && value > baseline * (1.0 + trigger)) {
                count++
            }
        }
        // Second pass: collect
        val out = IntArray(count)
        var j = 0
        for (i in 0 until n) {
            val baseline = data[idx(i, PortfolioCol.BASELINE)]
            val value    = data[idx(i, PortfolioCol.VALUE)]
            if (baseline > 0.0 && value > baseline * (1.0 + trigger)) {
                out[j++] = i
            }
        }
        return out
    }

    // ── rebalanceCandidates(): IntArray ────────────────────────────────────

    /**
     * Identify asset indices eligible for rebalance.
     *
     * An asset is a rebalance candidate when its value has fallen below
     * `baseline * (1 - trigger)`.
     *
     * @param trigger  Trigger fraction (e.g. 0.035 for 3.5 %). Defaults to
     *                 the Dreamer 1.2 genome default
     *                 [Genome.FLAT_REBALANCE_TRIGGER_PERCENT] = 0.035.
     * @return Sorted [IntArray] of candidate row indices.
     */
    fun rebalanceCandidates(trigger: Double = 0.035): IntArray {
        val n = assetCount
        // First pass: count
        var count = 0
        for (i in 0 until n) {
            val baseline = data[idx(i, PortfolioCol.BASELINE)]
            val value    = data[idx(i, PortfolioCol.VALUE)]
            if (baseline > 0.0 && value < baseline * (1.0 - trigger)) {
                count++
            }
        }
        // Second pass: collect
        val out = IntArray(count)
        var j = 0
        for (i in 0 until n) {
            val baseline = data[idx(i, PortfolioCol.BASELINE)]
            val value    = data[idx(i, PortfolioCol.VALUE)]
            if (baseline > 0.0 && value < baseline * (1.0 - trigger)) {
                out[j++] = i
            }
        }
        return out
    }

    // ── totalManagedBaseline(excludeIndices): Double ───────────────────────

    /**
     * Sum of baselines across all managed assets, optionally excluding
     * specific indices (e.g. assets already queued for harvest).
     *
     * Two-pass approach: the exclusion set is typically small, so a linear
     * scan is faster than a HashSet for the hot path.
     *
     * @param excludeIndices  Asset row indices to skip. Empty by default.
     * @return Sum of baselines for non-excluded assets with baseline > 0.
     */
    fun totalManagedBaseline(excludeIndices: IntArray = IntArray(0)): Double {
        val n = assetCount
        var sum = 0.0
        for (i in 0 until n) {
            // Check exclusion
            var excluded = false
            for (j in excludeIndices.indices) {
                if (excludeIndices[j] == i) {
                    excluded = true
                    break
                }
            }
            if (excluded) continue

            val baseline = data[idx(i, PortfolioCol.BASELINE)]
            if (baseline > 0.0) {
                sum += baseline
            }
        }
        return sum
    }

    // ── updateBaselines(dragCoeff): PortfolioTensor ────────────────────────

    /**
     * Update baselines using SPAR drag — matches Dreamer 1.2.js lines 778-789.
     *
     * JS formula:
     * ```
     * gap = baseline - value
     * newBaseline = value + (gap * dragCoeff)
     * ```
     * Symmetric — applies whether value is above or below baseline.
     * Only runs when both baseline > 0 and value > 0.
     */
    fun updateBaselines(dragCoeff: Double): PortfolioTensor {
        val n = assetCount
        val newData = data.copyOf()

        for (i in 0 until n) {
            val baseOffset = i * COLS
            val baseline   = newData[baseOffset + PortfolioCol.BASELINE]
            val value      = newData[baseOffset + PortfolioCol.VALUE]

            if (baseline > 0.0 && value > 0.0) {
                val gap = baseline - value
                val newBaseline = value + (gap * dragCoeff)
                newData[baseOffset + PortfolioCol.BASELINE] = newBaseline
            }

            // Refresh deviation for every asset
            val currentBaseline = newData[baseOffset + PortfolioCol.BASELINE]
            newData[baseOffset + PortfolioCol.DEVIATION] =
                if (currentBaseline > 0.0)
                    (value - currentBaseline) / currentBaseline
                else
                    0.0
        }

        return PortfolioTensor(symbols, newData)
    }

    // ── Symbol lookup ─────────────────────────────────────────────────────

    /**
     * Find the row index for a given [symbol], or `-1` if not found.
     *
     * Linear scan — suitable for the typical portfolio size of tens of assets.
     * For larger portfolios, callers should maintain an external
     * `Map<String, Int>` index.
     */
    fun indexOf(symbol: String): Int {
        for (i in symbols.indices) {
            if (symbols[i] == symbol) return i
        }
        return -1
    }

    // ── Factory ────────────────────────────────────────────────────────────

    companion object {
        /**
         * Create a [PortfolioTensor] from parallel arrays.
         *
         * Packs the per-asset scalars into the row-major [DoubleArray] and
         * pre-computes the DEVIATION column as `(value - baseline) / baseline`.
         *
         * @param symbols   Asset symbol strings.
         * @param prices    Current price per asset.
         * @param values    Current USD value per asset.
         * @param baselines Target baseline per asset.
         * @param quantities Raw quantity held per asset.
         * @return Fully initialized [PortfolioTensor].
         */
        fun fromArrays(
            symbols: Array<String>,
            prices: DoubleArray,
            values: DoubleArray,
            baselines: DoubleArray,
            quantities: DoubleArray,
        ): PortfolioTensor {
            val n = symbols.size
            require(prices.size == n)     { "prices.size ${prices.size} != $n" }
            require(values.size == n)     { "values.size ${values.size} != $n" }
            require(baselines.size == n)  { "baselines.size ${baselines.size} != $n" }
            require(quantities.size == n) { "quantities.size ${quantities.size} != $n" }

            val data = DoubleArray(n * COLS)
            for (i in 0 until n) {
                val base = i * COLS
                data[base + PortfolioCol.PRICE]     = prices[i]
                data[base + PortfolioCol.VALUE]     = values[i]
                data[base + PortfolioCol.BASELINE]  = baselines[i]
                data[base + PortfolioCol.QUANTITY]  = quantities[i]
                data[base + PortfolioCol.DEVIATION] =
                    if (baselines[i] > 0.0)
                        (values[i] - baselines[i]) / baselines[i]
                    else
                        0.0
            }
            return PortfolioTensor(symbols, data)
        }
    }

    // ── equals / hashCode ──────────────────────────────────────────────────

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PortfolioTensor) return false
        if (!symbols.contentEquals(other.symbols)) return false
        return data.contentEquals(other.data)
    }

    override fun hashCode(): Int {
        var result = symbols.contentHashCode()
        result = 31 * result + data.contentHashCode()
        return result
    }

    override fun toString(): String =
        "PortfolioTensor(assets=$assetCount, symbols=${symbols.contentToString()})"
}
