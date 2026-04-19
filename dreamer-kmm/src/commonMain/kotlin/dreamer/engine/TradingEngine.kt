/**
 * TradingEngine — stateful orchestrator for the Dreamer tick-cycle loop.
 *
 * Wires together all pure signal functions into a cohesive trading cycle.
 * No I/O occurs; trade records are returned in [CycleResult.trades] for
 * the caller to execute.  All state mutations produce new immutable objects.
 *
 * Algorithm mirrors Dreamer 1.2.js update() (lines 638–1370).
 *
 * @see dreamer.signal.evaluateHarvest
 * @see dreamer.signal.evaluateRebalance
 * @see dreamer.signal.evaluatePortfolioHarvest
 * @see dreamer.signal.allocateHarvestProceeds
 */
package dreamer.engine

import dreamer.genome.Genome
import dreamer.risk.RiskState
import dreamer.risk.updateRiskMetrics
import dreamer.regime.Regime
import dreamer.regime.classify
import dreamer.signal.AllocationParams
import dreamer.signal.AssetAllocationInfo
import dreamer.signal.HarvestParams
import dreamer.signal.HarvestResult
import dreamer.signal.HarvestType
import dreamer.signal.PortfolioAssetRow
import dreamer.signal.PortfolioHarvestParams
import dreamer.signal.PortfolioHarvestState
import dreamer.signal.RebalanceConfig
import dreamer.signal.RebalanceResult
import dreamer.signal.RebalanceState
import dreamer.signal.RebalanceType
import dreamer.signal.TrailingState
import dreamer.signal.allocateHarvestProceeds
import dreamer.signal.crashProtectionActive
import dreamer.signal.evaluateHarvest
import dreamer.signal.evaluatePortfolioHarvest
import dreamer.signal.evaluateRebalance
import dreamer.signal.portfolioDeviationPercent
import dreamer.tensor.PortfolioCol
import dreamer.tensor.PortfolioTensor

// ── Constants ──────────────────────────────────────────────────────────────

/** Rolling window for per-asset price history. */
private const val PRICE_HISTORY_WINDOW = 200

// ── Engine mode ────────────────────────────────────────────────────────────

/**
 * Whether the engine is trading with real funds or running as a shadow simulation.
 */
enum class EngineMode {
    LIVE,
    SHADOW,
}

// ── Data classes ───────────────────────────────────────────────────────────

/**
 * Immutable snapshot of all engine state carried across cycles.
 *
 * @property baselines            Per-asset baseline values, parallel to symbols.
 * @property trailingStates       Per-asset harvest trailing state.
 * @property rebalanceStates      Per-asset rebalance tracking state.
 * @property portfolioHarvestState  Portfolio-level harvest tracking state.
 * @property lastActionTimestamps   Epoch-millis of last trade per asset.
 * @property riskState            Peak / max-drawdown tracking.
 * @property cashBalance          Available cash (post-trade estimate).
 * @property lastTotalValue       Total portfolio value from the previous cycle.
 * @property lastCyclePrices      Prices from the previous cycle, parallel to symbols.
 * @property priceHistory         Rolling per-asset price history.
 * @property totalHarvested       Cumulative USD harvested across all cycles.
 * @property totalTrades          Cumulative trade count across all cycles.
 */
data class EngineState(
    val baselines: DoubleArray,
    val trailingStates: Map<String, TrailingState>,
    val rebalanceStates: Map<String, RebalanceState>,
    val portfolioHarvestState: PortfolioHarvestState,
    val lastActionTimestamps: Map<String, Long>,
    val riskState: RiskState,
    val cashBalance: Double,
    val lastTotalValue: Double,
    val lastCyclePrices: DoubleArray,
    val priceHistory: Map<String, DoubleArray>,
    val totalHarvested: Double,
    val totalTrades: Int,
)

/**
 * Result of a single trading cycle.
 *
 * @property anyTrades       Whether any trade records were generated.
 * @property stateChanged    Whether any engine state was modified.
 * @property harvestedAmount Total USD harvested this cycle (estimated).
 * @property updatedState    New immutable engine state for the next cycle.
 * @property trades          Ordered list of trade instructions for the caller.
 */
data class CycleResult(
    val anyTrades: Boolean,
    val stateChanged: Boolean,
    val harvestedAmount: Double,
    val updatedState: EngineState,
    val trades: List<TradeRecord>,
)

/**
 * A single trade instruction for the caller to execute.
 *
 * @property asset    Trading pair symbol (e.g. "BTC").
 * @property side     "BUY" or "SELL".
 * @property quantity Number of units to trade.
 * @property price    Estimated execution price.
 * @property note     Human-readable description of why the trade was generated.
 */
data class TradeRecord(
    val asset: String,
    val side: String,
    val quantity: Double,
    val price: Double,
    val note: String,
)

/**
 * Input snapshot of the current portfolio state.
 *
 * All arrays are parallel — element [i] refers to the same asset.
 *
 * @property symbols    Trading pair symbols.
 * @property prices     Current price per asset.
 * @property values     Current USD value per asset holding.
 * @property quantities Current quantity held per asset.
 */
data class PortfolioInput(
    val symbols: Array<String>,
    val prices: DoubleArray,
    val values: DoubleArray,
    val quantities: DoubleArray,
)

// ── Helpers ────────────────────────────────────────────────────────────────

/**
 * Append [price] to a rolling-window [history] array, keeping at most [window] elements.
 * Returns a new array; the input is never mutated.
 */
private fun appendToHistory(history: DoubleArray?, price: Double, window: Int): DoubleArray {
    if (price <= 0.0) return history ?: DoubleArray(0)
    if (history == null || history.isEmpty()) return doubleArrayOf(price)
    val len = history.size
    if (len < window) {
        val out = DoubleArray(len + 1)
        for (i in 0 until len) out[i] = history[i]
        out[len] = price
        return out
    }
    // Shift left, drop oldest
    val out = DoubleArray(len)
    for (i in 0 until len - 1) out[i] = history[i + 1]
    out[len - 1] = price
    return out
}

// ── Main cycle function ───────────────────────────────────────────────────

/**
 * Execute one tick-cycle of the Dreamer trading engine.
 *
 * Pure-functional — performs no I/O.  All trade decisions are returned as
 * [TradeRecord] entries in [CycleResult.trades] for the caller to execute.
 * The returned [CycleResult.updatedState] is a new immutable snapshot that
 * the caller should store for the next cycle.
 *
 * Algorithm (mirrors Dreamer 1.2.js update() lines 638–1370):
 *
 *  1.  Compute currentTotalValue = sum(values) + cashBalance
 *  2.  If LIVE: auto-init new assets (baseline = value for new assets)
 *  3.  If LIVE: detect cash extraction (>12% drop, stable prices → heal baselines, skip)
 *  4.  Update lastTotalValue and lastCyclePrices
 *  5.  Update risk metrics (peak / drawdown)
 *  6.  Build PortfolioTensor from input arrays + baselines
 *  7.  Compute portfolioDeviationPercent
 *  8.  SPAR drag updateBaselines on the tensor (skip in SHADOW)
 *  9.  RegimeDetector.classify with price history stats
 *  10. Portfolio harvest evaluation (if enabled)
 *  11. For each non-excluded asset (if portfolio harvest didn't execute):
 *      evaluate individual harvest, record trades
 *  12. Harvest proceeds allocation → reinvest / BTC / ETH instructions
 *  13. For each non-excluded asset (skipping harvest-flagged):
 *      evaluate rebalance, record trades
 *  14. Crash protection check
 *  15. Return CycleResult
 *
 * @param input               Current portfolio snapshot.
 * @param state               Engine state from the previous cycle.
 * @param genome              Evolutionary parameter pack.
 * @param nowMs               Current epoch time in milliseconds.
 * @param mode                LIVE or SHADOW mode.
 * @param excludeFromHarvest  Symbols to skip for harvest operations.
 * @param excludeFromRebalance Symbols to skip for rebalance operations.
 * @return [CycleResult] with trades and updated state.
 */
fun runCycle(
    input: PortfolioInput,
    state: EngineState,
    genome: Genome,
    nowMs: Long,
    mode: EngineMode,
    excludeFromHarvest: Set<String> = emptySet(),
    excludeFromRebalance: Set<String> = emptySet(),
): CycleResult {
    val n = input.symbols.size
    val trades = mutableListOf<TradeRecord>()
    var stateChanged = false
    var harvestedAmount = 0.0
    var totalTrades = state.totalTrades

    // Mutable working copies — packaged into immutable result at the end.
    var baselines = state.baselines.copyOf()
    val updatedTrailingStates = state.trailingStates.toMutableMap()
    val updatedRebalanceStates = state.rebalanceStates.toMutableMap()
    val updatedLastActionTimestamps = state.lastActionTimestamps.toMutableMap()
    var availableCash = state.cashBalance

    // ── Step 1: Compute currentTotalValue ───────────────────────────────

    var currentTotalValue = 0.0
    for (i in 0 until n) currentTotalValue += input.values[i]
    currentTotalValue += state.cashBalance

    // ── Step 2: Auto-init new assets (LIVE only) ────────────────────────
    //  (JS lines 664–675)

    if (mode == EngineMode.LIVE) {
        val (initChanged, initBaselines) = autoInitNewAssets(
            symbols = input.symbols,
            values = input.values,
            baselines = baselines,
        )
        if (initChanged) {
            baselines = initBaselines
            stateChanged = true
        }
    }

    // ── Step 3: Detect cash extraction (LIVE only) ──────────────────────
    //  (JS lines 678–710)

    if (mode == EngineMode.LIVE && state.lastTotalValue > 0.0) {
        val extraction = detectCashExtraction(
            lastTotalValue = state.lastTotalValue,
            currentTotalValue = currentTotalValue,
            symbols = input.symbols,
            currentPrices = input.prices,
            lastPrices = state.lastCyclePrices,
            currentValues = input.values,
            baselines = baselines,
        )
        if (extraction.skipTrading) {
            // Heal baselines, update tracking, skip trading.
            // JS line 705: this.lastTotalValue = currentTotalValue
            val newState = state.copy(
                baselines = extraction.healedBaselines,
                lastTotalValue = currentTotalValue,
            )
            return CycleResult(
                anyTrades = false,
                stateChanged = true,
                harvestedAmount = 0.0,
                updatedState = newState,
                trades = emptyList(),
            )
        }
    }

    // ── Step 4: Update tracking values ──────────────────────────────────
    //  (JS lines 712–714)

    val updatedLastTotalValue = currentTotalValue
    val updatedLastCyclePrices = input.prices.copyOf()

    // ── Step 5: Update risk metrics ─────────────────────────────────────
    //  (JS lines 717–722)

    val updatedRiskState = updateRiskMetrics(currentTotalValue, state.riskState)

    // ── Step 6: Build PortfolioTensor ───────────────────────────────────

    val tensor = PortfolioTensor.fromArrays(
        symbols = input.symbols,
        prices = input.prices,
        values = input.values,
        baselines = baselines,
        quantities = input.quantities,
    )

    // ── Step 7: Compute portfolioDeviationPercent ───────────────────────
    //  (JS lines 734–737 — uses REBALANCE_EXCLUDE for exclusion)

    val excludeIndicesList = mutableListOf<Int>()
    for (i in 0 until n) {
        if (input.symbols[i] in excludeFromRebalance) {
            excludeIndicesList.add(i)
        }
    }
    val excludeIndices = excludeIndicesList.toIntArray()

    val portDevPct = portfolioDeviationPercent(input.values, baselines, excludeIndices)

    // ── Step 8: SPAR drag (skip in SHADOW mode) ─────────────────────────
    //  (JS lines 775–789)

    if (mode != EngineMode.SHADOW) {
        val dragCoeff = genome.d(Genome.SPAR_DRAG_COEFFICIENT)
        val draggedTensor = tensor.updateBaselines(dragCoeff)
        val draggedBaselines = draggedTensor.column(PortfolioCol.BASELINE)
        for (i in 0 until n) baselines[i] = draggedBaselines[i]
    }

    // ── Step 9: Update price history & regime detection ─────────────────
    //  (JS lines 740–747 for price history)

    val updatedPriceHistory = HashMap<String, DoubleArray>(state.priceHistory)
    for (i in 0 until n) {
        val sym = input.symbols[i]
        val price = input.prices[i]
        if (price > 0.0) {
            updatedPriceHistory[sym] = appendToHistory(
                state.priceHistory[sym], price, PRICE_HISTORY_WINDOW
            )
        }
    }

    // Regime detection — classify using BTC price history if available,
    // otherwise first symbol with sufficient data.
    // Computed but not currently used for control flow in the trading algorithm.
    @Suppress("UNUSED_VARIABLE")
    val historyForRegime: DoubleArray = run {
        updatedPriceHistory["BTC"]
            ?: (if (n > 0) updatedPriceHistory[input.symbols[0]] else null)
            ?: DoubleArray(0)
    }
    val regime: Regime = classify(historyForRegime)

    // ── Step 10: Portfolio harvest evaluation ────────────────────────────
    //  (JS lines 814–888)

    val portHarvestParams = PortfolioHarvestParams(
        triggerDeviationPercent = genome.d(Genome.PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT),
        confirmationCycles = genome.d(Genome.PORTFOLIO_HARVEST_CONFIRMATION_CYCLES),
        minAssetSurplus = genome.d(Genome.MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST),
    )

    val portAssets = mutableListOf<PortfolioAssetRow>()
    for (i in 0 until n) {
        portAssets.add(
            PortfolioAssetRow(
                symbol = input.symbols[i],
                baseline = baselines[i],
                value = input.values[i],
                price = input.prices[i],
            )
        )
    }

    val portHarvestResult = evaluatePortfolioHarvest(
        enablePortfolioHarvest = genome.enablePortfolioHarvest,
        portfolioDeviationPercent = portDevPct,
        assets = portAssets,
        excludeSymbols = excludeFromHarvest,
        nowMs = nowMs,
        state = state.portfolioHarvestState,
        params = portHarvestParams,
    )

    var portfolioHarvestExecuted = false

    if (portHarvestResult.executed) {
        portfolioHarvestExecuted = true
        for (si in portHarvestResult.sells) {
            val sym = si.symbol
            val price = input.prices[si.symbolIndex]
            val soldValue = si.qtyToSell * price

            trades.add(
                TradeRecord(
                    asset = sym,
                    side = "SELL",
                    quantity = si.qtyToSell,
                    price = price,
                    note = "Portfolio Harvest",
                )
            )

            // JS line 855: tokenBaselines[sym] = originalBaseline (baseline stays)
            // Clear trailing state for this asset
            updatedTrailingStates.remove(sym)
            harvestedAmount += soldValue
            availableCash += soldValue
            updatedLastActionTimestamps[sym] = nowMs
            stateChanged = true
            totalTrades++
        }
    }

    // ── Step 11: Individual asset harvest ───────────────────────────────
    //  (JS lines 890–994)

    val harvestFlagged = mutableSetOf<String>()

    // Reusable genome buffer to avoid per-asset allocation
    val resolvedGenome = DoubleArray(Genome.WIDTH)

    if (!portfolioHarvestExecuted) {
        for (i in 0 until n) {
            val sym = input.symbols[i]
            if (sym in excludeFromHarvest) continue
            val baseline = baselines[i]
            if (baseline <= 0.0) continue

            // Resolve per-asset genome
            genome.resolveInto(sym, resolvedGenome)

            val harvestParams = HarvestParams(
                triggerPercent = resolvedGenome[Genome.FLAT_HARVEST_TRIGGER_PERCENT],
                takePercent = resolvedGenome[Genome.HARVEST_TAKE_PERCENT],
                cycleThreshold = resolvedGenome[Genome.HARVEST_CYCLE_THRESHOLD],
                minSurplus = resolvedGenome[Genome.MIN_SURPLUS_FOR_HARVEST],
                minSurplusForced = resolvedGenome[Genome.MIN_SURPLUS_FOR_FORCED_HARVEST],
                forcedTimeoutMs = resolvedGenome[Genome.FORCED_HARVEST_TIMEOUT_MS],
                targetAdjustPercent = resolvedGenome[Genome.TARGET_ADJUST_PERCENT],
            )

            val trailing = updatedTrailingStates[sym] ?: TrailingState()

            val harvestResult: HarvestResult = evaluateHarvest(
                totalValue = input.values[i],
                baseline = baseline,
                price = input.prices[i],
                nowMs = nowMs,
                state = trailing,
                params = harvestParams,
            )

            // Update trailing state in the map
            if (harvestResult.updatedState.flagged) {
                updatedTrailingStates[sym] = harvestResult.updatedState
            } else {
                updatedTrailingStates.remove(sym)
            }

            if (harvestResult.type != HarvestType.NONE && harvestResult.qtyToSell > 0.0) {
                val price = input.prices[i]
                val soldValue = harvestResult.qtyToSell * price

                trades.add(
                    TradeRecord(
                        asset = sym,
                        side = "SELL",
                        quantity = harvestResult.qtyToSell,
                        price = price,
                        note = "${harvestResult.type} Harvest",
                    )
                )

                // Update baseline to the new post-harvest baseline
                baselines[i] = harvestResult.newBaseline
                harvestedAmount += soldValue
                availableCash += soldValue
                updatedLastActionTimestamps[sym] = nowMs

                // Clear trailing state (already reset by evaluateHarvest)
                updatedTrailingStates.remove(sym)
                stateChanged = true
                totalTrades++
            } else if (harvestResult.updatedState.flagged) {
                // Track harvest-flagged assets so rebalance can skip them
                harvestFlagged.add(sym)
            }
        }
    }

    // ── Step 12: Harvest proceeds allocation ────────────────────────────
    //  (JS lines 996–1178)

    if (harvestedAmount > 0.0) {
        val allocParams = AllocationParams.fromGenome(genome.doubles)

        // Build asset allocation info from current state
        val allocAssets = mutableListOf<AssetAllocationInfo>()
        for (i in 0 until n) {
            allocAssets.add(
                AssetAllocationInfo(
                    symbol = input.symbols[i],
                    baseline = baselines[i],
                    value = input.values[i],
                    price = input.prices[i],
                )
            )
        }

        // Total asset value before harvest (pre-sell values)
        var totalAssetValueBeforeHarvest = 0.0
        for (i in 0 until n) totalAssetValueBeforeHarvest += input.values[i]

        val allocResult = allocateHarvestProceeds(
            harvestedAmount = harvestedAmount,
            cashBalance = availableCash,
            totalAssetValueBeforeHarvest = totalAssetValueBeforeHarvest,
            assets = allocAssets,
            excludeSymbols = excludeFromRebalance,
            params = allocParams,
        )

        // 12a. Build reinvest instructions
        for (instr in allocResult.reinvestInstructions) {
            val price = input.prices[instr.symbolIndex]
            val qty = if (price > 0.0) instr.buyUSD / price else 0.0
            if (qty > 0.0) {
                // 12b. Raise baseline by growth factor
                baselines[instr.symbolIndex] += instr.baselineIncrease

                trades.add(
                    TradeRecord(
                        asset = instr.symbol,
                        side = "BUY",
                        quantity = qty,
                        price = price,
                        note = "Growth Reinvest",
                    )
                )

                availableCash -= instr.buyUSD
                updatedLastActionTimestamps[instr.symbol] = nowMs
                // Clear any active rebalance state for reinvested asset
                updatedRebalanceStates.remove(instr.symbol)
                stateChanged = true
                totalTrades++
            }
        }

        // 12c. BTC buy instruction
        val btcBuy = allocResult.btcBuy
        if (btcBuy != null) {
            val btcIdx = input.symbols.indexOf("BTC")
            if (btcIdx >= 0 && input.prices[btcIdx] > 0.0) {
                val qty = btcBuy.buyUSD / input.prices[btcIdx]
                if (qty > 0.0) {
                    trades.add(
                        TradeRecord(
                            asset = "BTC",
                            side = "BUY",
                            quantity = qty,
                            price = input.prices[btcIdx],
                            note = "Allocated BTC Buy",
                        )
                    )
                    // JS line 1141: baseline = (currentQty + newQty) * price
                    baselines[btcIdx] = (input.quantities[btcIdx] + qty) * input.prices[btcIdx]
                    availableCash -= btcBuy.buyUSD
                    updatedLastActionTimestamps["BTC"] = nowMs
                    stateChanged = true
                    totalTrades++
                }
            }
        }

        // 12c. ETH buy instruction
        val ethBuy = allocResult.ethBuy
        if (ethBuy != null) {
            val ethIdx = input.symbols.indexOf("ETH")
            if (ethIdx >= 0 && input.prices[ethIdx] > 0.0) {
                val qty = ethBuy.buyUSD / input.prices[ethIdx]
                if (qty > 0.0) {
                    trades.add(
                        TradeRecord(
                            asset = "ETH",
                            side = "BUY",
                            quantity = qty,
                            price = input.prices[ethIdx],
                            note = "Allocated ETH Buy",
                        )
                    )
                    // JS line 1166: baseline = (currentQty + newQty) * price
                    baselines[ethIdx] = (input.quantities[ethIdx] + qty) * input.prices[ethIdx]
                    availableCash -= ethBuy.buyUSD
                    updatedLastActionTimestamps["ETH"] = nowMs
                    stateChanged = true
                    totalTrades++
                }
            }
        }
    }

    // ── Step 13: Rebalance ──────────────────────────────────────────────
    //  (JS lines 1181–1328)

    for (i in 0 until n) {
        val sym = input.symbols[i]
        // Skip excluded assets
        if (sym in excludeFromRebalance) continue
        // Skip assets currently flagged for harvest (JS line 1184)
        if (sym in harvestFlagged) continue
        val baseline = baselines[i]
        if (baseline <= 0.0) continue

        // Resolve per-asset genome
        genome.resolveInto(sym, resolvedGenome)
        val rebalanceConfig = RebalanceConfig.fromGenome(resolvedGenome)

        val rebState = updatedRebalanceStates[sym]

        val rebResult: RebalanceResult = evaluateRebalance(
            state = rebState,
            totalValueUSD = input.values[i],
            currentBaseline = baseline,
            currentPrice = input.prices[i],
            nowMs = nowMs,
            cashBalanceUSD = availableCash,
            config = rebalanceConfig,
        )

        // Update rebalance tracking state
        if (rebResult.updatedState != null) {
            updatedRebalanceStates[sym] = rebResult.updatedState
        } else {
            updatedRebalanceStates.remove(sym)
        }

        if (rebResult.type != RebalanceType.NONE && rebResult.buyQty > 0.0) {
            val price = input.prices[i]
            val buyCost = rebResult.buyQty * price

            trades.add(
                TradeRecord(
                    asset = sym,
                    side = "BUY",
                    quantity = rebResult.buyQty,
                    price = price,
                    note = "${rebResult.type} Rebalance",
                )
            )

            baselines[i] = rebResult.newBaseline
            availableCash -= buyCost
            updatedLastActionTimestamps[sym] = nowMs

            // Clear rebalance state (evaluateRebalance returns null when done)
            updatedRebalanceStates.remove(sym)
            stateChanged = true
            totalTrades++
        }
    }

    // ── Step 14: Crash protection check ─────────────────────────────────
    //  (JS lines 794–812)

    @Suppress("UNUSED_VARIABLE")
    val isCrashProtectionActive = if (genome.enableCrashProtection) {
        crashProtectionActive(
            values = input.values,
            baselines = baselines,
            triggerMinNegDevPercent = genome.d(Genome.CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT),
            triggerAssetPercent = genome.d(Genome.CP_TRIGGER_ASSET_PERCENT),
        )
    } else {
        false
    }

    // ── Step 15: Build and return result ────────────────────────────────

    // Clamp cash — should not go negative
    if (availableCash < 0.0) availableCash = 0.0

    val newState = EngineState(
        baselines = baselines,
        trailingStates = updatedTrailingStates.toMap(),
        rebalanceStates = updatedRebalanceStates.toMap(),
        portfolioHarvestState = portHarvestResult.updatedState,
        lastActionTimestamps = updatedLastActionTimestamps.toMap(),
        riskState = updatedRiskState,
        cashBalance = availableCash,
        lastTotalValue = updatedLastTotalValue,
        lastCyclePrices = updatedLastCyclePrices,
        priceHistory = updatedPriceHistory.toMap(),
        totalHarvested = state.totalHarvested + harvestedAmount,
        totalTrades = totalTrades,
    )

    return CycleResult(
        anyTrades = trades.isNotEmpty(),
        stateChanged = stateChanged,
        harvestedAmount = harvestedAmount,
        updatedState = newState,
        trades = trades.toList(),
    )
}
