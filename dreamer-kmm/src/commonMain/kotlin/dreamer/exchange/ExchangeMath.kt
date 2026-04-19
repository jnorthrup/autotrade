package dreamer.exchange

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Map of symbol -> minimum increment step for quantity rounding.
 * Mirrors the JS `minIncrementMap` used by roundQty.
 * Keys populated by caller; defaults applied when missing.
 */
typealias MinIncrementMap = Map<String, Double>

/**
 * Map of symbol -> minimum order quantity.
 * Mirrors the JS `MIN_ORDER_QTY_MAP` used by checkMinQuantity.
 */
typealias MinOrderQtyMap = Map<String, Double>

/**
 * Round a quantity down to the nearest valid increment for the symbol.
 * Port of Dreamer 1.2.js roundQty() lines 1426-1434.
 *
 * Algorithm:
 *  1. Look up step from [minIncrements]. If missing, use 1e-8 default (1e-4 for LTC).
 *  2. If qty < step/10, return "0.0".
 *  3. rounded = floor(qty / step) * step
 *  4. Format to appropriate decimal places (max 18, min 8).
 *  5. Strip trailing zeros.
 *  6. If result < step/10, return "0.0".
 *
 * Returns String to match JS behavior (avoids floating-point representation issues).
 */
fun roundQty(symbol: String, qty: Double, minIncrements: MinIncrementMap): String {
    val step = minIncrements[symbol]
    val safeStep = step ?: if (symbol == "LTC") 0.0001 else 0.00000001

    if (qty.isNaN() || qty < safeStep / 10.0) return "0.0"

    val rounded = floor(qty / safeStep) * safeStep
    val decimalPlaces = if (step != null && step.toString().contains('.')) {
        step.toString().split('.')[1].length
    } else {
        8
    }
    val displayPlaces = min(18, max(8, decimalPlaces))

    var str = rounded.toFixed(displayPlaces)
    // Strip trailing zeros after last non-zero digit: 1.2300 -> 1.23
    str = str.replace(Regex("\\.(\\d*[1-9])0+$"), ".$1")
    str = str.replace(Regex("\\.0+$"), "")

    return if (str.toDoubleOrNull()?.let { it < safeStep / 10.0 } == true) "0.0" else str
}

// Extension to format Double to fixed decimal places (commonMain-safe)
private fun Double.toFixed(decimals: Int): String {
    var s = ""
    var v = this
    if (v < 0) { s += "-"; v = -v }
    val intPart = v.toLong()
    s += intPart.toString()
    var frac = v - intPart.toDouble()
    if (decimals > 0) {
        s += "."
        for (i in 0 until decimals) {
            frac *= 10.0
            val digit = frac.toLong()
            s += digit.toString()
            frac -= digit.toDouble()
        }
    }
    return s
}

/** Minimum trade value in USD — mirrors JS MIN_TRADE_USD = 0.25 */
private const val MIN_TRADE_USD = 0.25

/**
 * Check if a trade's USD value exceeds the minimum dust threshold.
 * Port of Dreamer 1.2.js checkMinTrade() lines 1436-1443.
 */
fun checkMinTrade(usdValue: Double): Boolean = usdValue >= MIN_TRADE_USD

/**
 * Check if a quantity exceeds the minimum order quantity for a symbol.
 * Port of Dreamer 1.2.js checkMinQuantity() lines 1445-1451.
 */
fun checkMinQuantity(symbol: String, qty: Double, minOrderQty: MinOrderQtyMap): Boolean {
    val minQty = minOrderQty[symbol] ?: return true
    return qty >= minQty
}
