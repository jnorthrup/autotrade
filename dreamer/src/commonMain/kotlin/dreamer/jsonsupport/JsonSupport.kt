package dreamer.jsonsupport

import borg.trikeshed.parse.json.JsElement
import borg.trikeshed.parse.json.JsPath
import borg.trikeshed.parse.json.JsonSupport as TrikeShedJsonSupport

typealias JsIndex = borg.trikeshed.parse.json.JsIndex
typealias JsContext = borg.trikeshed.parse.json.JsContext

/**
 * Dreamer-facing JsonSupport facade.
 *
 * This object is intentionally tiny: it statically links against the
 * TrikeShed wasm/JVM JsonSupport implementation so Dreamer can keep
 * 64-bit-safe numeric deserialization at the edge of the shell.
 */
object JsonSupport {
    fun parse(text: String): Any? = TrikeShedJsonSupport.parse(text)

    fun index(
        text: String,
        depths: MutableList<Int>? = null,
        takeFirst: Int? = null,
    ): JsElement = TrikeShedJsonSupport.index(text, depths, takeFirst)

    fun pathOf(vararg steps: Any?): JsPath = TrikeShedJsonSupport.pathOf(*steps)

    fun query(
        text: String,
        path: JsPath,
        reifyResult: Boolean = true,
        depths: MutableList<Int>? = null,
    ): Any? = TrikeShedJsonSupport.query(text, path, reifyResult, depths)

    fun query(
        text: String,
        path: List<*>,
        reifyResult: Boolean = true,
        depths: MutableList<Int>? = null,
    ): Any? = TrikeShedJsonSupport.query(text, path, reifyResult, depths)

    fun query(
        text: String,
        query: String,
        reifyResult: Boolean = true,
        depths: MutableList<Int>? = null,
    ): Any? = TrikeShedJsonSupport.query(text, query, reifyResult, depths)
}
