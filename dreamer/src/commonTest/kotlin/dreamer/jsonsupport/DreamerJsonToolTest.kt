package dreamer.jsonsupport

import kotlin.test.Test
import kotlin.test.assertEquals

class DreamerJsonToolTest {
    private val json = """{"id64":5532807773,"coords":{"x":157.0,"y":-27,"z":-70},"bodies":[{"name":"Jackson's Lighthouse"},{"name":"Jackson's Lighthouse 1"}],"numbers":[1,2,3],"flag":true,"missing":null}"""

    @Test
    fun `tool renders scalar queries as canonical json`() {
        assertEquals("5532807773", DreamerJsonTool.queryJson(json, "id64"))
        assertEquals("157.0", DreamerJsonTool.queryJson(json, "coords.x"))
        assertEquals("\"Jackson's Lighthouse\"", DreamerJsonTool.queryJson(json, "bodies/0/name"))
        assertEquals("true", DreamerJsonTool.queryJson(json, "flag"))
        assertEquals("null", DreamerJsonTool.queryJson(json, "missing"))
    }

    @Test
    fun `tool renders object and array queries as canonical json`() {
        assertEquals("""{"x":157.0,"y":-27,"z":-70}""", DreamerJsonTool.queryJson(json, "coords"))
        assertEquals("""[1,2,3]""", DreamerJsonTool.queryJson(json, "numbers"))
        assertEquals("""["alpha",1,true,null]""", DreamerJsonTool.parseJson("""["alpha",1,true,null]"""))
    }

    @Test
    fun `tool reports stable value kinds`() {
        assertEquals("number", DreamerJsonTool.queryType(json, "id64"))
        assertEquals("object", DreamerJsonTool.queryType(json, "coords"))
        assertEquals("array", DreamerJsonTool.queryType(json, "numbers"))
        assertEquals("string", DreamerJsonTool.queryType(json, "bodies/0/name"))
        assertEquals("null", DreamerJsonTool.queryType(json, "missing"))
    }
}