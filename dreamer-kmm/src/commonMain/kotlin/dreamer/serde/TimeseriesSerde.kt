package dreamer.serde

import borg.trikeshed.lib.*

/**
 * A single price tick from market_data.jsonl.
 *
 * Wire format: `{"t":1234567890,"p":{"BTC":65000.12,"ETH":3400.56}}`
 */
data class Tick(
    val timestamp: Long,
    val prices: Map<String, Double>,
)

/**
 * Core state snapshot serialized to liveEngineState.json.
 */
data class EngineState(
    val baselines: Map<String, Double>,
    val trailingState: Map<String, TrailingStateSnapshot>,
    val lastActionTimestamps: Map<String, Long>,
    val peakTotalValue: Double,
    val maxDrawdownPercent: Double,
    val genomeDoubles: DoubleArray,
    val genomeOverrides: Map<String, DoubleArray>,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is EngineState) return false
        return baselines == other.baselines &&
            trailingState == other.trailingState &&
            lastActionTimestamps == other.lastActionTimestamps &&
            peakTotalValue == other.peakTotalValue &&
            maxDrawdownPercent == other.maxDrawdownPercent &&
            genomeDoubles.contentEquals(other.genomeDoubles) &&
            genomeOverrides == other.genomeOverrides
    }

    override fun hashCode(): Int {
        var result = baselines.hashCode()
        result = 31 * result + trailingState.hashCode()
        result = 31 * result + lastActionTimestamps.hashCode()
        result = 31 * result + peakTotalValue.hashCode()
        result = 31 * result + maxDrawdownPercent.hashCode()
        result = 31 * result + genomeDoubles.contentHashCode()
        result = 31 * result + genomeOverrides.hashCode()
        return result
    }
}

data class TrailingStateSnapshot(
    val flagged: Boolean,
    val harvestCycleCount: Int,
    val flaggedAt: Long?,
    val previousDeviation: Double?,
)

// ---------------------------------------------------------------------------
//  Tick serde — manual JSON via TrikeShed JsonParser for parse, StringBuilder for write
// ---------------------------------------------------------------------------

/**
 * Parse a single JSONL line into a [Tick].
 */
fun parseTick(jsonLine: String): Tick? {
    val src = jsonLine.trim()
    if (src.isEmpty()) return null
    try {
        val charSeries = src.toSeries()
        val root = borg.trikeshed.parse.json.JsonParser.reify(charSeries) as? Map<*, *> ?: return null

        val t = (root["t"] as? Number)?.toLong() ?: return null
        @Suppress("UNCHECKED_CAST")
        val pRaw = root["p"] as? Map<String, Any> ?: return null
        val prices = mutableMapOf<String, Double>()
        for (entry in pRaw.entries) {
            val v = entry.value
            if (v is Number && v.toDouble() > 0.0) prices[entry.key] = v.toDouble()
        }
        return Tick(t, prices)
    } catch (_: Exception) {
        return null
    }
}

/**
 * Parse an entire JSONL file content into ticks, taking the last [limit] entries.
 */
fun parseTimeseries(content: String, limit: Int = Int.MAX_VALUE): List<Tick> {
    val lines = content.trim().lines()
    val window = if (limit < lines.size) lines.takeLast(limit) else lines
    return window.mapNotNull { parseTick(it) }
}

/**
 * Serialize a [Tick] back to a JSONL string.
 * Manual StringBuilder — no reflection, commonMain-safe.
 */
fun serializeTick(tick: Tick): String = buildString {
    append("{\"t\":")
    append(tick.timestamp)
    append(",\"p\":{")
    val sortedKeys = tick.prices.keys.toList().sorted()
    var first = true
    for (sym in sortedKeys) {
        val price = tick.prices[sym]!!
        if (!first) append(',')
        first = false
        append('"')
        append(sym)
        append("\":")
        appendDouble(price)
    }
    append("}}")
}

// ---------------------------------------------------------------------------
//  Engine state serde
// ---------------------------------------------------------------------------

/**
 * Parse liveEngineState.json content into an [EngineState].
 */
fun parseEngineState(content: String): EngineState? {
    val src = content.trim()
    if (src.isEmpty()) return null
    try {
        val charSeries = src.toSeries()
        val root = borg.trikeshed.parse.json.JsonParser.reify(charSeries) as? Map<*, *> ?: return null

        // baselines: { "BTC": 65000.0, ... }
        val baselines = mutableMapOf<String, Double>()
        @Suppress("UNCHECKED_CAST")
        val bRaw = root["baselines"] as? Map<String, Any>
        if (bRaw != null) {
            for (entry in bRaw.entries) {
                (entry.value as? Number)?.toDouble()?.let { baselines[entry.key] = it }
            }
        }

        // trailingState: { "BTC": { flagged: bool, harvestCycleCount: int, ... }, ... }
        val trailing = mutableMapOf<String, TrailingStateSnapshot>()
        @Suppress("UNCHECKED_CAST")
        val tsRaw = root["trailingState"] as? Map<String, Any>
        if (tsRaw != null) {
            for (entry in tsRaw.entries) {
                val m = entry.value as? Map<String, Any> ?: continue
                trailing[entry.key] = TrailingStateSnapshot(
                    flagged = m["flagged"] as? Boolean ?: false,
                    harvestCycleCount = (m["harvestCycleCount"] as? Number)?.toInt() ?: 0,
                    flaggedAt = (m["flaggedAt"] as? Number)?.toLong(),
                    previousDeviation = (m["previousDeviation"] as? Number)?.toDouble(),
                )
            }
        }

        // lastActionTimestamps: { "BTC": 1234567890, ... }
        val lastAction = mutableMapOf<String, Long>()
        @Suppress("UNCHECKED_CAST")
        val laRaw = root["lastActionTimestamps"] as? Map<String, Any>
        if (laRaw != null) {
            for (entry in laRaw.entries) {
                (entry.value as? Number)?.toLong()?.let { lastAction[entry.key] = it }
            }
        }

        val peakTotalValue = (root["peakTotalValue"] as? Number)?.toDouble() ?: 0.0
        val maxDrawdownPercent = (root["maxDrawdownPercent"] as? Number)?.toDouble() ?: 0.0

        // Genome restoration is done by Genome companion; here we just pass through raw data
        val genomeDoubles = DoubleArray(0)
        val genomeOverrides = mutableMapOf<String, DoubleArray>()

        return EngineState(
            baselines = baselines,
            trailingState = trailing,
            lastActionTimestamps = lastAction,
            peakTotalValue = peakTotalValue,
            maxDrawdownPercent = maxDrawdownPercent,
            genomeDoubles = genomeDoubles,
            genomeOverrides = genomeOverrides,
        )
    } catch (_: Exception) {
        return null
    }
}

/**
 * Serialize an [EngineState] to JSON string.
 */
fun serializeEngineState(state: EngineState): String = buildString {
    append("{\"baselines\":{")
    val sortedBKeys = state.baselines.keys.toList().sorted()
    var first = true
    for (k in sortedBKeys) {
        val v = state.baselines[k]!!
        if (!first) append(',')
        first = false
        append('"').append(k).append("\":")
        appendDouble(v)
    }
    append("},\"trailingState\":{")
    val sortedTSKeys = state.trailingState.keys.toList().sorted()
    first = true
    for (sym in sortedTSKeys) {
        val ts = state.trailingState[sym]!!
        if (!first) append(',')
        first = false
        append('"').append(sym).append("\":{\"flagged\":")
        append(ts.flagged)
        append(",\"harvestCycleCount\":")
        append(ts.harvestCycleCount)
        append(",\"flaggedAt\":")
        append(ts.flaggedAt ?: "null")
        append(",\"previousDeviation\":")
        append(ts.previousDeviation ?: "null")
        append('}')
    }
    append("},\"lastActionTimestamps\":{")
    val sortedLAKeys = state.lastActionTimestamps.keys.toList().sorted()
    first = true
    for (k in sortedLAKeys) {
        val v = state.lastActionTimestamps[k]!!
        if (!first) append(',')
        first = false
        append('"').append(k).append("\":").append(v)
    }
    append("},\"peakTotalValue\":")
    appendDouble(state.peakTotalValue)
    append(",\"maxDrawdownPercent\":")
    appendDouble(state.maxDrawdownPercent)
    append('}')
}

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------

private fun StringBuilder.appendDouble(d: Double) {
    if (d == d.toLong().toDouble() && !d.isInfinite()) {
        append(d.toLong())
        append(".0")
    } else {
        append(d)
    }
}
