@file:Suppress("unused", "NON_EXPORTABLE_TYPE")
@file:OptIn(kotlin.js.ExperimentalJsExport::class)

package dreamer.platform

import dreamer.tensor.*
import dreamer.regime.*
import dreamer.genome.Genome
import dreamer.serde.Tick

/**
 * WASM-JS bridge for Dreamer — primitive-only exports.
 *
 * Kotlin/Wasm `@JsExport` restricts parameters and returns to
 * primitives (Int, Double, Boolean, String) only. No arrays, no objects.
 *
 * Strategy: JS glue allocates a SharedArrayBuffer / Float64Array view
 * and passes (offset, length) pairs. These bridge functions read/write
 * the wasm linear memory directly via a singleton Memory object that
 * the JS glue initializes before calling these exports.
 *
 * For full-array operations (EMA, batch), the JS side calls internal
 * (non-exported) Kotlin functions by reading/writing the shared memory
 * buffer directly through the wasm module's memory export.
 */

/** Shared linear memory — JS initializes this before calling exports. */
internal var sharedMemory: DoubleArray = DoubleArray(0)

// ── @JsExport bridge (pure primitives) ────────────────────────────────

/** Set the shared memory pointer. JS calls this once after instantiation. */
@JsExport
fun setMemory(baseAddr: Int): Unit {
    // baseAddr is ignored; we use the internal sharedMemory DoubleArray
    // which JS populates directly. This is a no-op placeholder.
}

/** Write a double into shared memory at offset. */
@JsExport
fun poke(offset: Int, value: Double): Unit {
    if (offset in sharedMemory.indices) sharedMemory[offset] = value
}

/** Read a double from shared memory at offset. */
@JsExport
fun peek(offset: Int): Double =
    if (offset in sharedMemory.indices) sharedMemory[offset] else Double.NaN

// ── Scalar computations (no memory access needed) ─────────────────────

/** EMA multiplier for given span. */
@JsExport
fun emaMultiplier(span: Int): Double = 2.0 / (span + 1.0)

/** Harvest deviation between totalValue and baseline. */
@JsExport
fun harvestDeviation(totalValue: Double, baseline: Double): Double =
    if (baseline > 0.0) (totalValue - baseline) / baseline else -1.0

/** Rebalance delta for single asset. */
@JsExport
fun rebalanceDelta(currentValue: Double, totalPortfolio: Double, targetWeight: Double): Double =
    totalPortfolio * targetWeight - currentValue

/** Default genome param by ordinal. */
@JsExport
fun defaultGenomeParam(ordinal: Int): Double =
    if (ordinal in 0 until Genome.WIDTH) Genome.DEFAULT.d(ordinal) else Double.NaN

/** Genome width constant. */
@JsExport
fun genomeWidth(): Int = Genome.WIDTH

/** Regime code for name string. */
@JsExport
fun regimeName(code: Int): String = when (code) {
    0 -> "UNKNOWN"
    1 -> "CRAB_CHOP"
    2 -> "BULL_RUSH"
    3 -> "BEAR_CRASH"
    4 -> "STEADY_GROWTH"
    5 -> "VOLATILE_CHOP"
    else -> "UNDEFINED"
}

// ── Array computations via sharedMemory ───────────────────────────────

/**
 * EMA: reads [srcOffset..srcOffset+len), writes result to [dstOffset..dstOffset+len).
 * Returns the last EMA value (useful for signal detection).
 */
@JsExport
fun emaToMemory(srcOffset: Int, dstOffset: Int, len: Int, span: Int): Double {
    val src = DoubleArray(len) { sharedMemory[srcOffset + it] }
    val result = dreamer.tensor.ema(src, span)
    for (i in 0 until len) {
        if (dstOffset + i < sharedMemory.size) sharedMemory[dstOffset + i] = result[i]
    }
    return result.last()
}

/**
 * Classify regime from prices in shared memory.
 * Returns regime code 0-5.
 */
@JsExport
fun classifyRegimeFromMemory(offset: Int, len: Int): Int {
    val prices = DoubleArray(len) { sharedMemory[offset + it] }
    return dreamer.regime.classify(prices).code
}

/**
 * Compute stdDev of values in shared memory.
 */
@JsExport
fun stdDevFromMemory(offset: Int, len: Int): Double {
    val arr = DoubleArray(len) { sharedMemory[offset + it] }
    return dreamer.tensor.stdDev(arr)
}

/**
 * Compute mean of values in shared memory.
 */
@JsExport
fun meanFromMemory(offset: Int, len: Int): Double {
    val arr = DoubleArray(len) { sharedMemory[offset + it] }
    return dreamer.tensor.mean(arr)
}

/**
 * EMA crossover signal from prices in memory.
 * Returns: 1.0 (bullish), -1.0 (bearish), 0.0 (neutral).
 */
@JsExport
fun emaSignal(srcOffset: Int, len: Int, shortPeriod: Int, longPeriod: Int): Double {
    val prices = DoubleArray(len) { sharedMemory[srcOffset + it] }
    val shortEma = dreamer.tensor.ema(prices, shortPeriod)
    val longEma = dreamer.tensor.ema(prices, longPeriod)
    if (shortEma.size < 2) return 0.0
    val n = shortEma.size - 1
    val prevDelta = shortEma[n - 1] - longEma[n - 1]
    val currDelta = shortEma[n] - longEma[n]
    return when {
        prevDelta <= 0.0 && currDelta > 0.0 -> 1.0
        prevDelta >= 0.0 && currDelta < 0.0 -> -1.0
        else -> 0.0
    }
}

/**
 * Regime stats from prices in memory.
 * Writes [roi, volatility, meanPrice] to dstOffset.
 * Returns 0 on success, -1 if insufficient data.
 */
@JsExport
fun regimeStatsToMemory(srcOffset: Int, srcLen: Int, dstOffset: Int): Int {
    val prices = DoubleArray(srcLen) { sharedMemory[srcOffset + it] }
    val detector = RegimeDetector()
    val stats = detector.stats(prices) ?: return -1
    if (dstOffset + 2 < sharedMemory.size) {
        sharedMemory[dstOffset] = stats.roi
        sharedMemory[dstOffset + 1] = stats.volatility
        sharedMemory[dstOffset + 2] = stats.meanPrice
    }
    return 0
}

// ── Exchange helpers (scalar, primitive-only) ──────────────────────────

/** Check if a USD trade value exceeds minimum dust threshold ($0.25). */
@JsExport
fun checkMinTradeValue(usdValue: Double): Boolean =
    dreamer.exchange.checkMinTrade(usdValue)

/** Check if a quantity meets minimum order size. Returns 1=true, 0=false. */
@JsExport
fun checkMinQty(symbol: String, qty: Double, minQty: Double): Int =
    if (dreamer.exchange.checkMinQuantity(symbol, qty, mapOf(symbol to minQty))) 1 else 0

/** Minimum trade USD constant ($0.25). */
@JsExport
fun minTradeUsd(): Double = 0.25
