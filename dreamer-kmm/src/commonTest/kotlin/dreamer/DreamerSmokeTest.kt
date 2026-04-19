package dreamer

import dreamer.tensor.*
import dreamer.genome.Genome
import dreamer.serde.*
import dreamer.regime.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class EmaTest {

    @Test
    fun ema_constantInput_returnsConstant() {
        val src = DoubleArray(20) { 100.0 }
        val result = ema(src, 10)
        // EMA of constant should be that constant
        assertTrue(result.all { it == 100.0 }, "EMA of constant should be constant")
    }

    @Test
    fun ema_risingInput_lastValueGreaterThanFirst() {
        val src = DoubleArray(50) { i -> 100.0 + i * 2.0 }
        val result = ema(src, 10)
        assertTrue(result.last() > result.first(), "Rising prices should produce rising EMA")
    }

    @Test
    fun ema_multiplier_isCorrect() {
        // Multiplier for span 10 = 2/(10+1) ≈ 0.1818
        val span = 10
        val mult = 2.0 / (span + 1)
        assertEquals(0.1818, mult, 0.001)
    }

    @Test
    fun mean_emptyArray_isNaN() {
        assertTrue(mean(DoubleArray(0)).isNaN())
    }

    @Test
    fun mean_knownValues() {
        assertEquals(3.0, mean(doubleArrayOf(2.0, 4.0, 3.0)), 0.0001)
    }

    @Test
    fun stdDev_knownValues() {
        val v = stdDev(doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0))
        assertEquals(2.0, v, 0.0001)
    }
}

class GenomeTest {

    @Test
    fun defaultGenome_hasCorrectWidth() {
        assertEquals(Genome.WIDTH, Genome.DEFAULT.doubles.size)
    }

    @Test
    fun defaultGenome_targetAdjustPercent() {
        // From Dreamer 1.2.js defaultGenome: ordinal 0 = 0.001
        assertEquals(0.001, Genome.DEFAULT.d(Genome.TARGET_ADJUST_PERCENT), 0.0001)
    }

    @Test
    fun genome_roundTripThroughDoubles() {
        val original = Genome.DEFAULT
        val reconstructed = Genome(original.doubles.copyOf())
        for (i in 0 until Genome.WIDTH) {
            assertEquals(original.d(i), reconstructed.d(i), "Ordinal $i mismatch")
        }
    }

    @Test
    fun genome_withOverride_resolvesCorrectly() {
        val overrides = mapOf("BTC" to DoubleArray(Genome.WIDTH) { i ->
            if (i == Genome.TARGET_ADJUST_PERCENT) 0.05 else Double.NaN
        })
        val g = Genome(Genome.DEFAULT.doubles.copyOf(), overrides = overrides)
        // BTC should use override
        assertEquals(0.05, g.d(Genome.TARGET_ADJUST_PERCENT, "BTC"), 0.0001)
        // ETH should fall back to default (0.001)
        assertEquals(0.001, g.d(Genome.TARGET_ADJUST_PERCENT, "ETH"), 0.0001)
    }
}

class TickSerdeTest {

    @Test
    fun roundTrip_singleAsset() {
        val tick = Tick(1700000000000L, mapOf("BTC" to 65000.50))
        val json = serializeTick(tick)
        val parsed = parseTick(json)
        assertNotNull(parsed)
        assertEquals(1700000000000L, parsed.timestamp)
        assertEquals(65000.50, parsed.prices["BTC"]!!, 0.01)
    }

    @Test
    fun roundTrip_multiAsset() {
        val tick = Tick(1700000000000L, mapOf("BTC" to 65000.0, "ETH" to 3400.0, "SOL" to 150.0))
        val json = serializeTick(tick)
        val parsed = parseTick(json)
        assertNotNull(parsed)
        assertEquals(3, parsed.prices.size)
        assertEquals(65000.0, parsed.prices["BTC"]!!, 0.01)
        assertEquals(3400.0, parsed.prices["ETH"]!!, 0.01)
        assertEquals(150.0, parsed.prices["SOL"]!!, 0.01)
    }

    @Test
    fun parseTick_invalidJson_returnsNull() {
        assertNull(parseTick(""))
        assertNull(parseTick("not json"))
        assertNull(parseTick("{}"))
    }

    @Test
    fun parseTimeseries_filtersBlankLines() {
        val content = """
            {"t":1,"p":{"BTC":100.0}}
            
            {"t":2,"p":{"BTC":200.0}}
        """.trimIndent()
        val ticks = parseTimeseries(content)
        assertEquals(2, ticks.size)
        assertEquals(100.0, ticks[0].prices["BTC"]!!, 0.01)
        assertEquals(200.0, ticks[1].prices["BTC"]!!, 0.01)
    }
}

class RegimeTest {

    @Test
    fun classify_steadyRise_isBullish() {
        // Steady rise: low volatility, positive ROI
        val prices = DoubleArray(100) { i -> 100.0 + i * 0.5 }
        val regime = classify(prices)
        assertTrue(regime == Regime.STEADY_GROWTH || regime == Regime.BULL_RUSH,
            "Steady rise should be STEADY_GROWTH or BULL_RUSH, got $regime")
    }

    @Test
    fun classify_crashDrop_isBear() {
        // Sharp drop: high negative ROI
        val prices = DoubleArray(100) { i -> 1000.0 - i * 5.0 }
        val regime = classify(prices)
        assertEquals(Regime.BEAR_CRASH, regime)
    }

    @Test
    fun classify_flat_isSideways() {
        // Flat with tiny noise
        val prices = DoubleArray(100) { i -> 100.0 + kotlin.math.sin(i.toDouble()) * 0.1 }
        val regime = classify(prices)
        assertTrue(regime == Regime.CRAB_CHOP || regime == Regime.STEADY_GROWTH,
            "Flat should be CRAB_CHOP or STEADY_GROWTH, got $regime")
    }

    @Test
    fun stats_insufficientData_returnsNull() {
        val detector = RegimeDetector()
        assertNull(detector.stats(DoubleArray(3) { 100.0 }))
    }
}

class DoubleArrayMathTest {

    @Test
    fun variance_knownValues() {
        val v = variance(doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0))
        assertEquals(4.0, v, 0.0001)
    }

    @Test
    fun windowSum_works() {
        val src = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = windowSum(src, 3)
        assertEquals(5, result.size)
        assertEquals(1.0, result[0], 0.001)  // [0..0] partial window: 1
        assertEquals(3.0, result[1], 0.001)  // [0..1] partial window: 1+2
        assertEquals(6.0, result[2], 0.001)  // [0..2] full window: 1+2+3
        assertEquals(9.0, result[3], 0.001)  // [1..3] full window: 2+3+4
        assertEquals(12.0, result[4], 0.001) // [2..4] full window: 3+4+5
    }

    @Test
    fun portfolioDeviation_zeroWhenMatched() {
        val vals = doubleArrayOf(100.0, 200.0)
        val bases = doubleArrayOf(100.0, 200.0)
        assertEquals(0.0, portfolioDeviation(vals, bases), 0.0001)
    }

    @Test
    fun portfolioDeviation_positiveWhenAbove() {
        val vals = doubleArrayOf(110.0, 220.0)
        val bases = doubleArrayOf(100.0, 200.0)
        assertTrue(portfolioDeviation(vals, bases) > 0.0)
    }
}
