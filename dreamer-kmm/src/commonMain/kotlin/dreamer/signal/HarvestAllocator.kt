/**
 * Dreamer 1.2 — Harvest Proceeds Allocation (pure-functional).
 *
 * Port of Dreamer 1.2.js lines 996-1178.
 *
 * Takes a harvested USD amount plus portfolio state and returns allocation
 * instructions for reinvestment, BTC buy, ETH buy, or cash retention.
 * No I/O — the caller is responsible for execution.
 */

package dreamer.signal

import kotlin.math.abs
import kotlin.math.min
import kotlin.math.pow

// ──────────────────────────────────────────────────────────────────────────────
// Allocation Mode
// ──────────────────────────────────────────────────────────────────────────────

enum class AllocationMode(val modifier: Double) {
    BALANCED(0.0),
    GROWTH(+0.15),
    DEFENSIVE(-0.15);

    companion object {
        /** Decode from the genome's raw double (0=BALANCED, 1=GROWTH, 2=DEFENSIVE). */
        fun fromRaw(raw: Double): AllocationMode =
            when (kotlin.math.round(raw).toInt()) {
                1 -> GROWTH
                2 -> DEFENSIVE
                else -> BALANCED
            }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Data classes
// ──────────────────────────────────────────────────────────────────────────────

data class ReinvestInstruction(
    /** Ordinal index of the asset in the portfolio list. */
    val symbolIndex: Int,
    /** Asset symbol, e.g. "LINK". */
    val symbol: String,
    /** Dollar amount to buy. */
    val buyUSD: Double,
    /** Amount to raise the asset's baseline by (buyUSD * growthFactor). */
    val baselineIncrease: Double
)

data class BuyInstruction(
    /** Asset symbol ("BTC" or "ETH"). */
    val symbol: String,
    /** Dollar amount to buy. */
    val buyUSD: Double
)

data class AllocationResult(
    val amountForReinvest: Double,
    val amountForBTC: Double,
    val amountForETH: Double,
    val amountKeptAsCash: Double,
    val reinvestInstructions: List<ReinvestInstruction>,
    val btcBuy: BuyInstruction?,
    val ethBuy: BuyInstruction?
) {
    companion object {
        /** Sentinel for "nothing to allocate". */
        val EMPTY = AllocationResult(
            amountForReinvest = 0.0,
            amountForBTC = 0.0,
            amountForETH = 0.0,
            amountKeptAsCash = 0.0,
            reinvestInstructions = emptyList(),
            btcBuy = null,
            ethBuy = null
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Input parameter pack
// ──────────────────────────────────────────────────────────────────────────────

data class AllocationParams(
    /** Cash reserve floor — if portfolio cash % is below this, keep 100 % harvest as cash. */
    val crashFundThreshold: Double,
    /** Minimum harvested amount worth allocating. Below this → keep as cash. */
    val minHarvestToAllocate: Double,
    /** Fraction of harvest directed to reinvestment (before mode modifier). */
    val reinvestPct: Double,
    /** Fraction of harvest directed to BTC buy. */
    val btcPct: Double,
    /** Fraction of harvest directed to ETH buy. */
    val ethPct: Double,
    /** BALANCED / GROWTH / DEFENSIVE — adjusts reinvestPct. */
    val allocationMode: AllocationMode,
    /** Minimum USD for a reinvestment buy to be emitted. */
    val minReinvestBuyUSD: Double,
    /** Fraction of buyUSD to add to baseline ("little less" growth logic). */
    val reinvestBaselineGrowthFactor: Double,
    /** Exponent for scoring dip depth (higher → stronger preference for deep dips). */
    val reinvestWeightExponent: Double,
    /** Maximum deviation to qualify as a reinvest candidate (negative, e.g. -0.01). */
    val minNegDevForReinvest: Double,
    /** Minimum USD for BTC buy instruction. */
    val minBtcBuyUSD: Double,
    /** Minimum USD for ETH buy instruction. */
    val minEthBuyUSD: Double
) {
    companion object {
        /**
         * Build [AllocationParams] from a [Genome] doubles array.
         * Accepts the raw DoubleArray so callers don't need the Genome import.
         */
        fun fromGenome(g: DoubleArray): AllocationParams = AllocationParams(
            crashFundThreshold          = g[14],
            minHarvestToAllocate        = g[15],
            reinvestPct                 = g[12],
            btcPct                      = g[10],
            ethPct                      = g[11],
            allocationMode              = AllocationMode.fromRaw(g[36]),
            minReinvestBuyUSD           = g[17],
            reinvestBaselineGrowthFactor = g[18],
            reinvestWeightExponent      = g[37],
            minNegDevForReinvest        = g[16],
            minBtcBuyUSD                = g[19],
            minEthBuyUSD                = g[20]
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Asset snapshot supplied by the caller
// ──────────────────────────────────────────────────────────────────────────────

data class AssetAllocationInfo(
    val symbol: String,
    val baseline: Double,
    /** Current USD value of the holding. */
    val value: Double,
    /** Current per-unit price (used by caller for qty math; kept here for completeness). */
    val price: Double
)

// ──────────────────────────────────────────────────────────────────────────────
// Main allocator
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Pure-functional harvest-allocator.
 *
 * @param harvestedAmount     USD obtained from the harvest sell.
 * @param cashBalance         Portfolio cash balance *including* the harvest proceeds.
 * @param totalAssetValueBeforeHarvest  Sum of all asset USD values *before* the sell.
 * @param assets              Snapshot of every asset (used for reinvest candidate scoring).
 * @param excludeSymbols      Symbols to skip when looking for reinvest candidates
 *                            (e.g. BTC, ETH which have their own allocation bucket).
 * @param params              Allocation parameters (typically derived from Genome).
 * @return                    [AllocationResult] with allocation instructions.
 */
fun allocateHarvestProceeds(
    harvestedAmount: Double,
    cashBalance: Double,
    totalAssetValueBeforeHarvest: Double,
    assets: List<AssetAllocationInfo>,
    excludeSymbols: Set<String>,
    params: AllocationParams
): AllocationResult {

    // ── Guard: nothing harvested ───────────────────────────────────────────
    if (harvestedAmount == 0.0) return AllocationResult.EMPTY

    // ── Compute current portfolio state ────────────────────────────────────
    // totalAssetValueBeforeHarvest includes the asset we just sold at its
    // pre-sell value.  Subtract the harvest amount and add the new cash balance.
    val currentTotalPortfolioValue =
        totalAssetValueBeforeHarvest - harvestedAmount + cashBalance
    val currentCashPercent =
        cashBalance / maxOf(1.0, currentTotalPortfolioValue)

    // ── CRASH FUND CHECK ───────────────────────────────────────────────────
    if (currentCashPercent < params.crashFundThreshold) {
        return AllocationResult(
            amountForReinvest = 0.0,
            amountForBTC = 0.0,
            amountForETH = 0.0,
            amountKeptAsCash = harvestedAmount,
            reinvestInstructions = emptyList(),
            btcBuy = null,
            ethBuy = null
        )
    }

    // ── Minimum threshold ──────────────────────────────────────────────────
    if (harvestedAmount < params.minHarvestToAllocate) {
        return AllocationResult(
            amountForReinvest = 0.0,
            amountForBTC = 0.0,
            amountForETH = 0.0,
            amountKeptAsCash = harvestedAmount,
            reinvestInstructions = emptyList(),
            btcBuy = null,
            ethBuy = null
        )
    }

    // ── Percentage allocation with mode modifier ───────────────────────────
    var reinvestPct = params.reinvestPct + params.allocationMode.modifier
    val btcPct = params.btcPct
    val ethPct = params.ethPct

    // Clamp reinvest to [0, 1]
    reinvestPct = reinvestPct.coerceIn(0.0, 1.0)

    // Normalize: if total > 100 %, cap reinvest (BTC/ETH are prioritised)
    val fixedAlloc = btcPct + ethPct
    if (reinvestPct + fixedAlloc > 1.0) {
        reinvestPct = maxOf(0.0, 1.0 - fixedAlloc)
    }

    // Dollar amounts
    val amountForReinvest = harvestedAmount * reinvestPct
    val amountForBTC = harvestedAmount * btcPct
    val amountForETH = harvestedAmount * ethPct
    val amountKeptAsCash = harvestedAmount - amountForReinvest - amountForBTC - amountForETH

    // ── Reinvestment: pick top-3 deepest dips ──────────────────────────────
    val reinvestInstructions = buildReinvestInstructions(
        amountForReinvest = amountForReinvest,
        assets = assets,
        excludeSymbols = excludeSymbols,
        exponent = params.reinvestWeightExponent,
        minNegDevForReinvest = params.minNegDevForReinvest,
        minReinvestBuyUSD = params.minReinvestBuyUSD,
        baselineGrowthFactor = params.reinvestBaselineGrowthFactor
    )

    // ── BTC / ETH buys ─────────────────────────────────────────────────────
    val btcBuy = if (amountForBTC >= params.minBtcBuyUSD) {
        BuyInstruction(symbol = "BTC", buyUSD = amountForBTC)
    } else null

    val ethBuy = if (amountForETH >= params.minEthBuyUSD) {
        BuyInstruction(symbol = "ETH", buyUSD = amountForETH)
    } else null

    return AllocationResult(
        amountForReinvest = amountForReinvest,
        amountForBTC = amountForBTC,
        amountForETH = amountForETH,
        amountKeptAsCash = amountKeptAsCash,
        reinvestInstructions = reinvestInstructions,
        btcBuy = btcBuy,
        ethBuy = ethBuy
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Reinvestment scoring engine
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Identify up to 3 deepest-under-baseline assets and emit reinvest instructions.
 *
 * JS algorithm (lines 1058-1128):
 *  1. Filter: baseline > 0, value < baseline, deviation <= MIN_NEG_DEVIATION
 *  2. Score = |deviation|^exponent — higher = deeper dip
 *  3. Sort descending, take top 3
 *  4. perCandidate = total / count
 *  5. buyUSD = min(perCandidate, gap)  where gap = baseline - value
 *  6. Emit if buyUSD >= min threshold; baselineIncrease = buyUSD * growthFactor
 */
private fun buildReinvestInstructions(
    amountForReinvest: Double,
    assets: List<AssetAllocationInfo>,
    excludeSymbols: Set<String>,
    exponent: Double,
    minNegDevForReinvest: Double,
    minReinvestBuyUSD: Double,
    baselineGrowthFactor: Double
): List<ReinvestInstruction> {
    if (amountForReinvest <= 0.0) return emptyList()

    // Scored candidates
    data class Candidate(
        val index: Int,
        val symbol: String,
        val baseline: Double,
        val value: Double,
        val deviation: Double,
        val score: Double
    )

    val candidates = mutableListOf<Candidate>()

    for ((idx, a) in assets.withIndex()) {
        if (a.symbol in excludeSymbols) continue
        if (a.baseline <= 0.0) continue
        if (a.value >= a.baseline) continue

        val deviation = (a.value - a.baseline) / a.baseline   // negative
        if (deviation > minNegDevForReinvest) continue          // not deep enough

        val score = abs(deviation).pow(exponent)
        candidates.add(Candidate(idx, a.symbol, a.baseline, a.value, deviation, score))
    }

    // Sort deepest dips first, take top 3
    candidates.sortByDescending { it.score }
    if (candidates.size > 3) {
        (candidates as MutableList).subList(3, candidates.size).clear()
    }

    if (candidates.isEmpty()) return emptyList()

    val perCandidate = amountForReinvest / candidates.size.toDouble()
    val instructions = mutableListOf<ReinvestInstruction>()

    for (c in candidates) {
        val gap = c.baseline - c.value
        val buyUSD = min(perCandidate, gap)

        if (buyUSD >= minReinvestBuyUSD) {
            val baselineIncrease = buyUSD * baselineGrowthFactor
            instructions.add(
                ReinvestInstruction(
                    symbolIndex = c.index,
                    symbol = c.symbol,
                    buyUSD = buyUSD,
                    baselineIncrease = baselineIncrease
                )
            )
        }
    }

    return instructions
}
