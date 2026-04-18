@file:OptIn(ExperimentalJsExport::class)

package dreamer.jsonsupport

import kotlin.js.ExperimentalJsExport
import kotlin.js.JsExport

@JsExport
fun dreamerJsonParse(text: String): String = DreamerJsonTool.parseJson(text)

@JsExport
fun dreamerJsonQuery(text: String, query: String): String = DreamerJsonTool.queryJson(text, query)

@JsExport
fun dreamerJsonQueryType(text: String, query: String): String = DreamerJsonTool.queryType(text, query)