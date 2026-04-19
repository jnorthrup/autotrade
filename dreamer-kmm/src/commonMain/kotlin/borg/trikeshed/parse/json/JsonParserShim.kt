package borg.trikeshed.parse.json

import borg.trikeshed.lib.*

/**
 * Minimal JsonParser shim for dreamer-kmm standalone compilation.
 * Provides `reify()` for JSON parsing without external dependencies.
 * Identical surface to TrikeShed's JsonParser — swapped out via composite build when available.
 */
object JsonParser {
    /**
     * Reify a JSON string into a Kotlin object tree.
     * Returns Map<String,Any?> for objects, List<Any?> for arrays, or primitives.
     */
    fun reify(src: Series<Char>): Any? {
        val s = src.toList().let { chars ->
            var start = 0
            var end = chars.size
            while (start < end && chars[start] in setOf(' ', '\t', '\n', '\r')) start++
            while (end > start && chars[end - 1] in setOf(' ', '\t', '\n', '\r')) end--
            chars.subList(start, end)
        }
        if (s.isEmpty()) return null
        return parseValue(s, 0).value
    }

    private data class ParseResult(val value: Any?, val nextIndex: Int)

    private fun parseValue(chars: List<Char>, index: Int): ParseResult {
        var i = skipWhitespace(chars, index)
        if (i >= chars.size) return ParseResult(null, i)
        return when (chars[i]) {
            '{' -> parseObject(chars, i)
            '[' -> parseArray(chars, i)
            '"' -> parseString(chars, i)
            't' -> ParseResult(true, i + 4)  // true
            'f' -> ParseResult(false, i + 5) // false
            'n' -> ParseResult(null, i + 4)   // null
            else -> parseNumber(chars, i)
        }
    }

    private fun parseObject(chars: List<Char>, index: Int): ParseResult {
        var i = skipWhitespace(chars, index)
        i++ // skip '{'
        val map = mutableMapOf<String, Any?>()
        i = skipWhitespace(chars, i)
        if (i < chars.size && chars[i] == '}') return ParseResult(map, i + 1)
        while (i < chars.size) {
            i = skipWhitespace(chars, i)
            val keyResult = parseString(chars, i)
            val key = keyResult.value as String
            i = skipWhitespace(chars, keyResult.nextIndex)
            if (i < chars.size && chars[i] == ':') i++
            i = skipWhitespace(chars, i)
            val valueResult = parseValue(chars, i)
            map[key] = valueResult.value
            i = skipWhitespace(chars, valueResult.nextIndex)
            if (i < chars.size && chars[i] == ',') { i++; continue }
            break
        }
        if (i < chars.size && chars[i] == '}') i++
        return ParseResult(map, i)
    }

    private fun parseArray(chars: List<Char>, index: Int): ParseResult {
        var i = skipWhitespace(chars, index)
        i++ // skip '['
        val list = mutableListOf<Any?>()
        i = skipWhitespace(chars, i)
        if (i < chars.size && chars[i] == ']') return ParseResult(list, i + 1)
        while (i < chars.size) {
            i = skipWhitespace(chars, i)
            val result = parseValue(chars, i)
            list.add(result.value)
            i = skipWhitespace(chars, result.nextIndex)
            if (i < chars.size && chars[i] == ',') { i++; continue }
            break
        }
        if (i < chars.size && chars[i] == ']') i++
        return ParseResult(list, i)
    }

    private fun parseString(chars: List<Char>, index: Int): ParseResult {
        var i = index
        if (chars[i] == '"') i++
        val sb = StringBuilder()
        while (i < chars.size) {
            when (chars[i]) {
                '\\' -> {
                    i++
                    if (i < chars.size) when (chars[i]) {
                        '"' -> sb.append('"')
                        '\\' -> sb.append('\\')
                        'n' -> sb.append('\n')
                        't' -> sb.append('\t')
                        'r' -> sb.append('\r')
                        '/' -> sb.append('/')
                        else -> sb.append(chars[i])
                    }
                    i++
                }
                '"' -> return ParseResult(sb.toString(), i + 1)
                else -> sb.append(chars[i++])
            }
        }
        return ParseResult(sb.toString(), i)
    }

    private fun parseNumber(chars: List<Char>, index: Int): ParseResult {
        var i = index
        val start = i
        if (i < chars.size && (chars[i] == '-' || chars[i] == '+')) i++
        while (i < chars.size && chars[i].isDigit()) i++
        var isFloat = false
        if (i < chars.size && chars[i] == '.') { isFloat = true; i++; while (i < chars.size && chars[i].isDigit()) i++ }
        if (i < chars.size && (chars[i] == 'e' || chars[i] == 'E')) {
            isFloat = true; i++
            if (i < chars.size && (chars[i] == '+' || chars[i] == '-')) i++
            while (i < chars.size && chars[i].isDigit()) i++
        }
        val text = chars.subList(start, i).joinToString("")
        val value: Any? = if (isFloat) text.toDoubleOrNull() else {
            val l = text.toLongOrNull()
            if (l != null && l in Int.MIN_VALUE..Int.MAX_VALUE) l.toInt() else l
        }
        return ParseResult(value, i)
    }

    private fun skipWhitespace(chars: List<Char>, index: Int): Int {
        var i = index
        while (i < chars.size && chars[i] in setOf(' ', '\t', '\n', '\r')) i++
        return i
    }
}
