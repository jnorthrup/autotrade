package dreamer.jsonsupport

import borg.trikeshed.lib.Join
import borg.trikeshed.lib.Series
import borg.trikeshed.lib.get
import borg.trikeshed.lib.size

/**
 * Standalone Dreamer JSON tool surface.
 *
 * The tool delegates parsing and path lookup to the statically linked
 * TrikeShed JsonSupport implementation, then re-encodes the reified result
 * into canonical JSON text so JS/Wasm consumers only need a String boundary.
 */
object DreamerJsonTool {
    fun parseJson(text: String): String = DreamerJsonValueCodec.toJson(JsonSupport.parse(text))

    fun queryJson(text: String, query: String): String = DreamerJsonValueCodec.toJson(JsonSupport.query(text, query))

    fun queryType(text: String, query: String): String = DreamerJsonValueCodec.typeOf(JsonSupport.query(text, query))
}

internal object DreamerJsonValueCodec {
    fun toJson(value: Any?): String = buildString {
        appendJsonValue(this, value)
    }

    fun typeOf(value: Any?): String = when {
        value == null -> "null"
        value is String || value is Char -> "string"
        value is Boolean -> "boolean"
        value is Number -> "number"
        value is Map<*, *> -> "object"
        asSeries(value) != null -> "array"
        value is Iterable<*> || value is Array<*> -> "array"
        else -> "unknown"
    }

    private fun appendJsonValue(builder: StringBuilder, value: Any?) {
        when (value) {
            null -> builder.append("null")
            is String -> appendJsonString(builder, value)
            is Char -> appendJsonString(builder, value.toString())
            is Boolean -> builder.append(if (value) "true" else "false")
            is Byte, is Short, is Int, is Long,
            is UByte, is UShort, is UInt, is ULong -> builder.append(value.toString())
            is Float -> appendFiniteDouble(builder, value.toDouble())
            is Double -> appendFiniteDouble(builder, value)
            is Map<*, *> -> appendJsonObject(builder, value)
            is Array<*> -> appendJsonArray(builder, value.asIterable())
            is Iterable<*> -> appendJsonArray(builder, value)
            else -> asSeries(value)?.let {
                appendJsonSeries(builder, it)
            } ?: appendJsonString(builder, value.toString())
        }
    }

    private fun appendFiniteDouble(builder: StringBuilder, value: Double) {
        if (value.isFinite()) {
            builder.append(value.toString())
        } else {
            builder.append("null")
        }
    }

    private fun appendJsonObject(builder: StringBuilder, map: Map<*, *>) {
        builder.append('{')
        var first = true
        for ((key, value) in map) {
            if (!first) {
                builder.append(',')
            }
            first = false
            appendJsonString(builder, key?.toString() ?: "null")
            builder.append(':')
            appendJsonValue(builder, value)
        }
        builder.append('}')
    }

    private fun appendJsonSeries(builder: StringBuilder, series: Series<Any?>) {
        builder.append('[')
        for (index in 0 until series.size) {
            if (index > 0) {
                builder.append(',')
            }
            appendJsonValue(builder, series[index])
        }
        builder.append(']')
    }

    private fun appendJsonArray(builder: StringBuilder, values: Iterable<*>) {
        builder.append('[')
        var first = true
        for (value in values) {
            if (!first) {
                builder.append(',')
            }
            first = false
            appendJsonValue(builder, value)
        }
        builder.append(']')
    }

    private fun appendJsonString(builder: StringBuilder, value: String) {
        builder.append('"')
        value.forEach { ch ->
            when (ch) {
                '\\' -> builder.append("\\\\")
                '"' -> builder.append("\\\"")
                '\b' -> builder.append("\\b")
                '\u000C' -> builder.append("\\f")
                '\n' -> builder.append("\\n")
                '\r' -> builder.append("\\r")
                '\t' -> builder.append("\\t")
                else -> if (ch.code < 0x20) {
                    builder.append("\\u")
                    builder.append(ch.code.toString(16).padStart(4, '0'))
                } else {
                    builder.append(ch)
                }
            }
        }
        builder.append('"')
    }

    @Suppress("UNCHECKED_CAST")
    private fun asSeries(value: Any?): Series<Any?>? = when {
        value is Join<*, *> && value.a is Int && value.b is Function1<*, *> -> value as Series<Any?>
        else -> null
    }
}