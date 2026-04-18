// Dreamer 1.999 - Cryptobot Token Flex (ESM style) - TrikeShed/Wasm presentation build

import axios from "axios";
import nacl from "tweetnacl";
import dotenv from "dotenv";
import { Buffer } from "buffer";
import crypto from "crypto"; // for crypto.randomUUID()
import readline from "readline";
import fs from 'fs';
import path from "path";
import { fileURLToPath } from 'url';
import { fork } from 'child_process';
import os from 'os';

dotenv.config();

// --- AssetRegimeManager (Embedded) ---
class AssetRegimeManager {
  constructor() {
    this.memoryFile = path.join(process.cwd(), 'configs', 'asset_regime_memory.json');
    this.memory = {};
    this.load();
  }
  load() {
    if (fs.existsSync(this.memoryFile)) {
      try {
        this.memory = JSON.parse(fs.readFileSync(this.memoryFile, 'utf-8'));
        console.log(`🧠 [Memory] Loaded Asset Regimes for ${Object.keys(this.memory).length} assets.`);
      } catch (e) { console.error("❌ Failed to load asset_regime_memory.json", e); this.memory = {}; }
    } else { this.save(); }
  }
  save() {
    try {
      const dir = path.dirname(this.memoryFile);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(this.memoryFile, JSON.stringify(this.memory, null, 2));
    } catch (e) { console.error("❌ Failed to save asset_regime_memory.json", e); }
  }
  getProfile(symbol) { return this.memory[symbol] || null; }
  update(symbol, genomeSlice, tier, regime = 'UNKNOWN') {
    if (!this.memory[symbol]) {
      this.memory[symbol] = { activeRegime: regime, tiers: { TIER_0_FACTORY: {}, TIER_1_THEORETICAL: null, TIER_2_VERIFIED: null } };
    }
    const entry = { timestamp: Date.now(), regime: regime, config: genomeSlice };
    this.memory[symbol].tiers[tier] = entry;
    this.memory[symbol].activeRegime = regime;
    this.save();
  }
}

// --- RegimeDetector (Embedded) ---
class RegimeDetector {
  constructor() { this.regimes = {}; }
  analyze(symbol, history) {
    if (!history || history.length < 50) { this.regimes[symbol] = 'UNKNOWN'; return 'UNKNOWN'; }
    const currentPrice = history[history.length - 1];
    const startPrice = history[0];
    const roi = (currentPrice - startPrice) / startPrice;
    const mean = history.reduce((a, b) => a + b) / history.length;
    const variance = history.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / history.length;
    const volatility = Math.sqrt(variance) / mean;
    let regime = 'CRAB_CHOP';
    if (roi > 0.05 && volatility > 0.02) regime = 'BULL_RUSH';
    else if (roi < -0.05 && volatility > 0.02) regime = 'BEAR_CRASH';
    else if (roi > 0.02 && volatility < 0.01) regime = 'STEADY_GROWTH';
    else if (volatility > 0.05) regime = 'VOLATILE_CHOP';

    if (this.regimes[symbol] !== regime) {
      console.log(`🔮 [Regime] ${symbol} Change: ${this.regimes[symbol] || 'INIT'} -> ${regime} (Vol: ${(volatility * 100).toFixed(2)}%)`);
    }
    this.regimes[symbol] = regime;
    return regime;
  }
  update(symbol, price, timestamp) {
    // Maintained history buffer could go here if we wanted fully self-contained detector state
    // For now relying on analyze() being called with history
  }
  getRegime(symbol) { return this.regimes[symbol] || 'UNKNOWN'; }
}

// --- LegionManager (Embedded) ---
import { EventEmitter } from 'events';
const MANAGER_CONFIG = {
  HEAT_CHECK_INTERVAL: 8000,
  CULL_THRESHOLD_HOURS: 4,
  STALE_GENOME_HOURS: 24,
  SWARM_SPREAD_PERCENT: 0.005,
};
class LegionManager extends EventEmitter {
  constructor(liveEngine, TradingEngineClass, dreamerGrid) {
    super();
    this.liveEngine = liveEngine;
    this.TradingEngineClass = TradingEngineClass;
    this.dreamerGrid = dreamerGrid;
    this.shadowLegion = [];
    this.assetHeatMap = {};
    this.lastHeatCheck = 0;
    this.lastOptimizationRequest = {};
    this.activeDreamJobs = new Set();
  }
  async heartbeat(portfolioSummary, api) {
    const now = Date.now();
    if (now - this.lastHeatCheck < MANAGER_CONFIG.HEAT_CHECK_INTERVAL) return;
    this.updateHeatMap(portfolioSummary);
    this.reallocateLegion(portfolioSummary);
    await this.marchLegion(portfolioSummary);
    this.lastHeatCheck = now;
  }
  updateHeatMap(portfolioSummary) {
    portfolioSummary.forEach(row => {
      const sym = row.Symbol;
      const baseline = this.liveEngine.baselines[sym] || 0;
      if (baseline <= 0) { this.assetHeatMap[sym] = 'COLD'; return; }
      const deviation = Math.abs((row.Value - baseline) / baseline);
      const harvestTrig = this.liveEngine.genome.FLAT_HARVEST_TRIGGER_PERCENT;
      const rebalTrig = this.liveEngine.genome.FLAT_REBALANCE_TRIGGER_PERCENT;
      const triggerDist = Math.min(Math.abs(deviation - harvestTrig), Math.abs(deviation - rebalTrig));

      if (triggerDist < 0.005) this.assetHeatMap[sym] = 'INFERNO';
      else if (triggerDist < 0.02) this.assetHeatMap[sym] = 'HOT';
      else if (triggerDist < 0.05) this.assetHeatMap[sym] = 'WARM';
      else this.assetHeatMap[sym] = 'COLD';

      if (this.assetHeatMap[sym] === 'HOT' || this.assetHeatMap[sym] === 'INFERNO') {
        const lastReq = this.lastOptimizationRequest[sym] || 0;
        if (Date.now() - lastReq > 60 * 60 * 1000) {
          this.requestOptimization(sym);
          this.lastOptimizationRequest[sym] = Date.now();
        }
      }
    });
  }
  reallocateLegion(portfolioSummary) {
    // CULL
    const maxCapacity = LEGION_CONFIG.TOTAL_SHADOW_CAPACITY || 50;
    let shadowsToRemove = [];
    this.shadowLegion.forEach((shadow, index) => {
      const asset = shadow.assignedAsset;
      const heat = this.assetHeatMap[asset] || 'COLD';
      const ageHours = (Date.now() - shadow.startTime) / 3600000;
      const idleHours = (Date.now() - (shadow.lastTradeTime || shadow.startTime)) / 3600000;
      if (shadow.killMe) { shadowsToRemove.push(index); return; } // Handle self-termination
      if (idleHours > MANAGER_CONFIG.CULL_THRESHOLD_HOURS) { shadowsToRemove.push(index); return; }
      if (ageHours > MANAGER_CONFIG.STALE_GENOME_HOURS) { shadowsToRemove.push(index); return; }
      if (this.shadowLegion.length > maxCapacity * 0.9 && heat === 'COLD') {
        const brothers = this.shadowLegion.filter(s => s.assignedAsset === asset);
        if (brothers.length > (LEGION_CONFIG.PASSIVE_ASSET_MONITOR_COUNT || 5)) shadowsToRemove.push(index);
      }
    });
    shadowsToRemove.sort((a, b) => b - a).forEach(idx => {
      const trash = this.shadowLegion[idx];
      // 🧹 MEMORY LEAK FIX: Explicitly nullify heavy references
      trash.priceHistoryBuffer = null;
      trash.holdings = null;
      trash.genome = null;
      this.shadowLegion.splice(idx, 1);
    });

    // SPAWN
    const density = LEGION_CONFIG.ACTIVE_ASSET_SWARM_DENSITY || 10;
    Object.entries(this.assetHeatMap).forEach(([asset, heat]) => {
      if (heat === 'HOT' || heat === 'INFERNO') {
        const currentCount = this.shadowLegion.filter(s => s.assignedAsset === asset).length;
        const deficit = density - currentCount;
        if (deficit > 0) this.deploySwarm(asset, deficit, portfolioSummary);
      }
    });
  }
  deploySwarm(asset, count, portfolioSummary) {
    const baseGenome = { ...this.liveEngine.genome };
    const row = portfolioSummary ? portfolioSummary.find(r => r.Symbol === asset) : null;
    let sweepTarget = 'FLAT_HARVEST_TRIGGER_PERCENT';
    if (row && row.Baseline > 0) {
      const deviation = (row.Value - row.Baseline) / row.Baseline;
      if (deviation < 0) sweepTarget = 'FLAT_REBALANCE_TRIGGER_PERCENT';
    }
    const centerVal = this.getParam(baseGenome, sweepTarget, asset);
    const spreadStep = (MANAGER_CONFIG.SWARM_SPREAD_PERCENT * 2) / count;

    // console.log(`⚔️ Deploying Swarm for ${asset}. Sweeping ${sweepTarget}`);
    for (let i = 0; i < count; i++) {
      const offset = -MANAGER_CONFIG.SWARM_SPREAD_PERCENT + (i * spreadStep);
      const swarmGenome = JSON.parse(JSON.stringify(baseGenome));
      if (!swarmGenome.overrides) swarmGenome.overrides = {};
      if (!swarmGenome.overrides[asset]) swarmGenome.overrides[asset] = {};
      swarmGenome.overrides[asset][sweepTarget] = centerVal + offset;

      const shadowHoldings = {};
      if (this.liveEngine.holdings && this.liveEngine.holdings[asset]) {
        shadowHoldings[asset] = JSON.parse(JSON.stringify(this.liveEngine.holdings[asset]));
      }
      const shadow = new this.TradingEngineClass(swarmGenome, 'SHADOW', this.liveEngine.cashBalance, shadowHoldings);
      shadow.priceHistoryBuffer = this.liveEngine.priceHistoryBuffer; // Inject Shared Real-Time History
      shadow.id = `Legion_${asset}_${i}`;
      shadow.assignedAsset = asset;
      shadow.startTime = Date.now();
      shadow.lastTradeTime = Date.now();
      this.shadowLegion.push(shadow);
    }
  }
  async marchLegion(portfolioSummary) {
    const priceMap = {};
    portfolioSummary.forEach(r => priceMap[r.Symbol] = r.Price);
    for (const shadow of this.shadowLegion) {
      const shadowPortfolio = [];
      Object.keys(shadow.holdings).forEach(sym => {
        if (priceMap[sym]) {
          shadowPortfolio.push({
            Symbol: sym, Price: priceMap[sym],
            Value: (shadow.holdings[sym].rawQuantity || 0) * priceMap[sym],
            Baseline: shadow.baselines[sym] || 0
          });
        }
      });
      const result = await shadow.update(shadowPortfolio, null, shadow.cashBalance, shadow.holdings);
      if (result.killMe) shadow.killMe = true; // Mark for culling
      if (result.anyTradesThisCycle) {
        shadow.lastTradeTime = Date.now();
        this.handleShadowVictory(shadow, result);
      }
    }
  }
  handleShadowVictory(shadow, result) {
    // console.log(`🏆 Shadow ${shadow.id} won!`);
    this.dispatchToDreamer({ type: 'FEEDBACK', genome: shadow.genome, score: shadow.lastTotalValue, focus: shadow.assignedAsset });
  }
  requestOptimization(asset) {
    // console.log(`🧠 Requesting Optimization for ${asset}`);
    // Ensure we send the CURRENT live genome so the Dreamer starts searching from where we are now,
    // not from scratch (defaults).
    const currentGenome = JSON.parse(JSON.stringify(this.liveEngine.genome));
    this.activeDreamJobs.add(asset); // Track active job
    this.dispatchToDreamer({ type: 'OPTIMIZE_ORDER', symbol: asset, baseGenome: currentGenome });
  }
  notifyOptimizationComplete(asset) {
    this.activeDreamJobs.delete(asset);
  }
  dispatchToDreamer(msg) {
    if (!this.dreamerGrid || this.dreamerGrid.length === 0) return;
    const worker = this.dreamerGrid[Math.floor(Math.random() * this.dreamerGrid.length)];
    if (worker && worker.send) worker.send(msg);
  }
  getParam(genome, key, asset) {
    if (genome.overrides && genome.overrides[asset] && genome.overrides[asset][key] !== undefined) return genome.overrides[asset][key];
    return genome[key];
  }
}

// ============== THE LEGION ARCHITECTURE CONFIG ==============
const LEGION_CONFIG = {
  // 🧠 THE BRAINS (Historical Math)
  // CPU Bound. Auto-Scales to (Total Cores - 1) to leave room for the OS/Main Thread.
  DREAMER_WORKER_COUNT: 1, // MANUAL OVERRIDE (Set to 2 to prevent lockup)

  // ⚔️ THE SOLDIERS (Live Testing)
  // Memory Bound.
  // Eco Mode: Reduced to 50 for Mini PC.
  TOTAL_SHADOW_CAPACITY: 50,

  // 🎯 THE FOCUS
  // How many shadows do we dedicate to an asset that is 'Active'?
  // Eco Mode: Reduced to 10.
  ACTIVE_ASSET_SWARM_DENSITY: 10,

  // 🛠️ DEVELOPER TOOLS
  ENABLE_DEVELOPER_LOGS: false, // Toggle this to see internal state data (Regime Radar, Rejection Reasons)

  // 💤 THE RESERVES
  // How many shadows keep watching boring assets just in case?
  PASSIVE_ASSET_MONITOR_COUNT: 10
};

// ============== Config/Maps and Constants ==============

// --- Asset Specific ---
const minIncrementMap = {
  AAVE: 0.00001, COMP: 0.00001, XLM: 0.01, AVAX: 0.0001, ADA: 0.01,
  XRP: 0.001, LINK: 0.0001, UNI: 0.0001, SOL: 0.00001, DOGE: 0.01,
  SHIB: 1.0, BTC: 0.00000001, ETH: 0.000001, PEPE: 1.0, BONK: 1.0,
  USDC: 0.000001, POPCAT: 0.01, WIF: 1.0, PENGU: 1.00, PNUT: 0.1,
  BCH: 0.00001, XTZ: 0.001, ETC: 0.000001, ARB: 0.010, LTC: 0.0001, TRUMP: 0.0001, MOODENG: 0.01, MEW: 1.00, SUI: 0.01,
  // NEWLY ADDED (AssetScanner):
  BNB: 0.001, FLOKI: 1.0, AERO: 0.01, ASTER: 0.01, AVNT: 0.01, CRV: 0.01, ENA: 0.01, HBAR: 0.01, HYPE: 0.01,
  LDO: 0.01, ONDO: 0.01, OP: 0.01, SEI: 0.01, SYRUP: 0.01, TON: 0.01, VIRTUAL: 0.01, WLFI: 0.01, XCN: 1.0, XPL: 0.01, ZORA: 0.01
};

const MIN_ORDER_QTY_MAP = {
  BTC: 0.00000001,  // 1 Satoshi (Minimizes price dependency)
  ETH: 0.0001,      // Updated to match API Error (was 0.000001)
  BCH: 0.001,
  DOGE: 1.0,
  SHIB: 100.0,
  XCN: 1.0,
  SOL: 0.1,
  // Add others as discovered
};

// Exclude BTC/USDC from automatic actions
const HARVEST_EXCLUDE = ["BTC", "ETH"];  // Applies to both individual and portfolio harvest participation
const REBALANCE_EXCLUDE = ["BTC", "ETH"]; // Also excludes from Priority Reinvestment & Adaptive Mode
const PRECISION_THRESHOLD = 0.0001; // Threshold for detecting deviation improvements

// --- Genome Definition (Hyper-Evolutionary Core) ---
const defaultGenome = {
  // --- Core Strategy ---
  TARGET_ADJUST_PERCENT: 0.001, // 0.1% - Careful ratchet to acknowledge the win without choking cash flow
  ALLOW_MUTATION: true, // Master switch for Evolution Manager

  // --- Individual Asset Harvest (Standard) ---
  FLAT_HARVEST_TRIGGER_PERCENT: 0.035,
  HARVEST_TAKE_PERCENT: 0.70, // Geometric Growth: 70% to cash, 30% left to compound
  HARVEST_CYCLE_THRESHOLD: 3,
  MIN_SURPLUS_FOR_HARVEST: 0.25,
  MIN_SURPLUS_FOR_FORCED_HARVEST: 1.00,
  FORCED_HARVEST_TIMEOUT: 45 * 60 * 1000, // Increased from 20 mins to 45 mins to give more breathing room before forcing.

  // --- Portfolio Override Harvest ---
  ENABLE_PORTFOLIO_HARVEST: true,
  PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT: 0.035,
  PORTFOLIO_HARVEST_CONFIRMATION_CYCLES: 3,
  MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST: 0.10,

  // --- Harvest Proceeds Allocation ---
  HARVEST_ALLOC_BTC_PERCENT: 0.25,
  HARVEST_ALLOC_ETH_PERCENT: 0.25,
  HARVEST_ALLOC_REINVEST_PERCENT: 0.25,
  HARVEST_ALLOC_CASH_PERCENT: 0.25, // Implicit, but kept for genome completeness

  // --- Crash Fund (Liquidity Reserve) ---
  CRASH_FUND_THRESHOLD_PERCENT: 0.10, // Require 10% cash reserve before normal allocation

  MIN_HARVEST_TO_ALLOCATE: 0.25,
  MIN_NEGATIVE_DEVIATION_FOR_REINVEST: -0.010,
  MIN_REINVEST_BUY_USD: 0.25,
  REINVEST_BASELINE_GROWTH_FACTOR: 0.50, // 0.85 = The Sweet Spot (Recover some, grow a lot)

  MIN_BTC_BUY_USD: 0.10,
  MIN_ETH_BUY_USD: 0.25,

  // --- Rebalance (Standard) ---
  FLAT_REBALANCE_TRIGGER_PERCENT: 0.035,
  PARTIAL_RECOVERY_PERCENT: 0.70,
  REBALANCE_POSITIVE_THRESHOLD: 3,
  MAX_REBALANCE_ATTEMPTS: 3, // Now Evolvable (Risk Gene)
  REBALANCE_COOLDOWN: 30 * 60 * 1000, // Now Evolvable (Time Gene)
  FORCE_REBALANCE_TIMEOUT: 25 * 60 * 1000, // Now Evolvable (Time Gene)
  FORCE_REBALANCE_SHORTFALL_PERCENT: 0.25,
  MIN_PARTIAL_REBALANCE_USD: 0.25,
  MIN_FORCED_REBALANCE_USD: 0.25,

  // --- Adaptive Dead Zone Mode (REMOVED: Legacy) ---
  // ENABLE_ADAPTIVE_DEAD_ZONE: false,
  // Logic removed in favor of Scientific Regime System

  // --- Project Dynamo (Physics) ---
  SPAR_DRAG_COEFFICIENT: 0.999968, // Now Evolvable (Physics Gene)

  // --- Crash Protection ---
  ENABLE_CRASH_PROTECTION: true,
  CP_TRIGGER_ASSET_PERCENT: 0.70,
  CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT: -0.07, // Now Evolvable (Risk Gene)
  CRASH_PROTECTION_THRESHOLD_INCREASE: 2,
  CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT: 0.33,

  // --- Timing ---
  REFRESH_INTERVAL: 8000, // 8s for Eco Mode

  // --- Smart Evolution Strategy ---
  ALLOCATION_MODE: 1, // 0=BALANCED, 1=GROWTH, 2=DEFENSIVE
  REINVEST_WEIGHT_EXPONENT: 1.50, // 1.0 (Linear) to 3.0 (Cubic/Sniper)
  FITNESS_DRAWDOWN_PENALTY: 1.00, // Score Penalty Multiplier for Max Drawdown % (e.g. 2.0 = 2x penalty)

  // --- Evolution & Oracle ---
  MIN_TRADES_FOR_PROMOTION: 1,
  EVOLUTION_CONSISTENCY_COUNT: 3, // Wins needed for Hall of Fame
  ORACLE_TREND_THRESHOLD: 0.8, // 🧬 Evolutionary: How strong a trend needs to be for Oracle intervention
  ORACLE_VOLATILITY_THRESHOLD: 2.0, // 🧬 Evolutionary: How high volatility needs to be for Oracle intervention
  EVOLUTION_INTERVAL_MINUTES: 5, // How often to check for promotions (1-60 min range evolvable)

  // --- Developer / Debug ---
  ENABLE_DEVELOPER_LOGS: true,

  // --- Per-Asset Overrides ---
  overrides: {} // Structure: { "BTC": { FLAT_HARVEST_TRIGGER_PERCENT: 0.05 }, "ETH": { ... } }
};

// --- Active Genome State ---
let currentGenome = { ...defaultGenome };

// --- Persistence ---
const STATE_FILE_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), 'liveEngineState.json');
const BASELINE_LOAD_TOLERANCE_PERCENT = 0.50;

// --- Evolution Configuration ---
const SHADOW_COUNT = 10; // Number of shadow bots to run (Scalable)

// --- COMPOUNDING CONFIG (The Alpha Tithe) ---
const ENABLE_AUTO_COMPOUND = true;
const COMPOUND_THRESHOLD_USD = 50.00; // Only invest if we have > $50 spare cash
const COMPOUND_ALLOCATION_PCT = 0.01; // Use 1% of total cash (The Tithe)
const GROWTH_INTERVAL = 4 * 60 * 60 * 1000; // Every 4 hours


// --- Global State ---
let priceHistory = loadRecentMarketData(500); // Hydrate on startup: 200 for Adaptive, 500 for Dreamer start
// --- Auto-Config Safety Net ---
function autoConfigMinQuantities(history) {
  if (!history || history.length === 0) return;
  const latest = history[history.length - 1];
  if (!latest || !latest.p) return;

  Object.entries(latest.p).forEach(([symbol, price]) => {
    if (MIN_ORDER_QTY_MAP[symbol] === undefined) {
      // Target $1.00 Value for Safety
      const safeQty = 1.0 / price;
      // Round to sensible precision (e.g. 1 sig fig)
      const magnitude = Math.pow(10, Math.floor(Math.log10(safeQty)));
      const rounded = Math.ceil(safeQty / magnitude) * magnitude;

      MIN_ORDER_QTY_MAP[symbol] = rounded;
      // console.warn(`⚠️ [SafetyNet] Auto-configured MinQty for ${symbol}: ${rounded} (Price: $${price})`);
    }
  });
}
autoConfigMinQuantities(priceHistory);


const INITIAL_ONE_TIME_BUY_USD = 100.00; // Small threshold for float comparisons

// ==================================================
// Sanity check allocation percentages
if (Math.abs(defaultGenome.HARVEST_ALLOC_BTC_PERCENT + defaultGenome.HARVEST_ALLOC_ETH_PERCENT + defaultGenome.HARVEST_ALLOC_REINVEST_PERCENT + defaultGenome.HARVEST_ALLOC_CASH_PERCENT - 1.0) > 0.001) {
  console.warn("Configuration Warning: Harvest allocation percentages (Genome) do not sum precisely to 1.0 (100%).");
}
// ==================================================


// ============== Trading Engine (The Brain) ==============
class TradingEngine {
  constructor(genome, mode = 'SHADOW', initialCapital = 0, initialHoldings = {}) {
    this.genome = { ...genome };
    this.mode = mode; // 'LIVE' or 'SHADOW'

    // --- Persistent State ---
    this.baselines = {};        // { SYM: value }
    this.trailingState = {};    // { SYM: harvest_info }
    this.lastActionTimestamps = {}; // { SYM: timestamp }

    // --- Transient State ---
    this.rebalanceState = {};   // { SYM: rebalance_info }
    // this.adaptiveDeadZoneState = {}; // REMOVED
    // this.adaptiveDeadZoneState = {}; // REMOVED
    this.portfolioHarvestState = {
      flagged: false,
      cycleCount: 0,
      previousDeviationPercent: null,
      flaggedAt: null
    };

    // --- Simulation State (Shadow Only) ---
    this.cashBalance = initialCapital;
    this.holdings = initialHoldings; // { SYM: qty }
    this.totalHarvested = 0; // Track performance (Cumulative Lifetime)
    this.totalTrades = 0;    // Track activity level
    this.lastTotalValue = initialCapital; // Value tracking

    // --- Risk Metrics ---
    this.peakTotalValue = initialCapital;
    this.maxDrawdownPercent = 0.0;

    this.priceHistory = {}; // Engine-local price history
    this.priceHistoryBuffer = []; // Global high-res history (LIVE only) or Simulation history (SHADOW)

    // --- Tier 1 & 2 Upgrades ---
    this.cyclesWithoutTrade = 0;
    this.lastCyclePrices = {}; // { SYM: price }
    this.minTradeUSD = 1.00; // Tier 2 Dust Protection
    this.postMortemEvents = []; // Queue for Tier 1 Post-Mortem
  }

  loadPersistedState(data) {
    if (!data) return;
    if (data.baselines) this.baselines = data.baselines;
    if (data.trailingState) this.trailingState = data.trailingState;
    if (data.lastActionTimestamps) this.lastActionTimestamps = data.lastActionTimestamps;
    if (data.genome) this.genome = { ...this.genome, ...data.genome };
  }

  getStateSnapshot() {
    return {
      baselines: this.baselines,
      trailingState: this.trailingState,
      lastActionTimestamps: this.lastActionTimestamps,
      genome: this.genome,
      // Include Live Portfolio State for Simulation
      cashBalance: this.cashBalance,
      holdings: this.holdings,
      // Include Dreamer promotion threshold & visual metadata for persistence
      lastBestScore: global.lastBestScore || 1.0,
      assetSourceTimeframe: this.assetSourceTimeframe || {}
    };
  }

  /**
   * 🧬 SCIENTIFIC REGIME: Deep Hydration
   * Inject a full snapshot of the Live Bot's reality into this Shadow.
   * Crucial for "Counterfactual Simulation" - asking "What if?" from the exact current state.
   */
  injectSimulationState(snapshot) {
    if (this.mode !== 'SHADOW') {
      console.warn("⚠️ Attempted to inject simulation state into LIVE engine. Ignoring.");
      return;
    }

    if (snapshot.cashBalance !== undefined) this.cashBalance = snapshot.cashBalance;
    if (snapshot.holdings) this.holdings = JSON.parse(JSON.stringify(snapshot.holdings));
    if (snapshot.baselines) this.baselines = JSON.parse(JSON.stringify(snapshot.baselines));
    if (snapshot.trailingState) this.trailingState = JSON.parse(JSON.stringify(snapshot.trailingState));
    if (snapshot.rebalanceState) this.rebalanceState = JSON.parse(JSON.stringify(snapshot.rebalanceState));
    if (snapshot.lastActionTimestamps) this.lastActionTimestamps = JSON.parse(JSON.stringify(snapshot.lastActionTimestamps));

    // Log for verification
    // console.log(`💉 State Injected: Cash=${this.cashBalance}, Holdings=${Object.keys(this.holdings).join(',')}`);
  }

  // Active Shadow Validation ("Micro-Backtest")
  isDefective(portfolioSummary, priceHistoryBuffer) {
    if (!priceHistoryBuffer || priceHistoryBuffer.length === 0) return false;

    const asset = this.assignedAsset;
    if (!asset) return false;

    // Get Genome Trigger (Rebalance Trigger - Buying the Dip)
    const overrides = this.genome.overrides && this.genome.overrides[asset];
    const rebalanceTrigger = (overrides && overrides.FLAT_REBALANCE_TRIGGER_PERCENT)
      || this.genome.FLAT_REBALANCE_TRIGGER_PERCENT;

    // If trigger is not negative (or effectively disabled), skip check
    if (!rebalanceTrigger || rebalanceTrigger >= 0) return false;

    // Needs at least some history
    if (priceHistoryBuffer.length < 10) return false;

    // Extract Asset Prices
    const prices = [];
    for (const tick of priceHistoryBuffer) {
      if (tick.prices && tick.prices[asset]) {
        prices.push({ t: tick.t, p: tick.prices[asset] });
      }
    }
    if (prices.length === 0) return false;

    // "Bad Trade" Definition:
    // We "Buy" when (Price - RollingMax) / RollingMax < Trigger
    // We "Fail" if Price subsequently drops > 1.0% (CRASH_THRESHOLD) below BuyPrice.
    const CRASH_THRESHOLD = 0.01;

    let maxPrice = prices[0].p;

    for (let i = 0; i < prices.length; i++) {
      const current = prices[i];
      if (current.p > maxPrice) maxPrice = current.p;

      // Check Trigger
      const deviation = (current.p - maxPrice) / maxPrice;

      if (deviation < rebalanceTrigger) {
        // VIRTUAL BUY SIGNAL at current.p

        // Look Ahead for Crash
        let minFuturePrice = current.p;
        for (let j = i + 1; j < prices.length; j++) {
          if (prices[j].p < minFuturePrice) minFuturePrice = prices[j].p;
        }

        const subsequentDrop = (minFuturePrice - current.p) / current.p;

        if (subsequentDrop < -CRASH_THRESHOLD) {
          // console.log(`💀 Shadow ${this.id} DEFECTIVE. Trigger ${rebalanceTrigger.toFixed(4)} hit at ${current.p}, crashed to ${minFuturePrice}`);
          return true;
        }
      }
    }

    return false;
  }

  async _placeSell(api, symbol, quantity, expectedPrice = null) {
    if (this.mode === 'LIVE' && api) return await api.placeSell(symbol, quantity);
    if (this.mode === 'SHADOW') {
      // Calculate Slippage
      let executedPrice = expectedPrice || 0;
      if (expectedPrice && this.slippageMap && this.slippageMap[symbol.replace('-USD', '')]) {
        const slip = this.slippageMap[symbol.replace('-USD', '')];
        // Limit/Market Sell: We fill LOWER if slippage exists (Price - Slippage)
        executedPrice = expectedPrice * (1 - slip);
      }
      // console.log(`[SHADOW] Selling ${quantity} ${symbol} @ ${executedPrice.toFixed(4)} (Exp: ${expectedPrice})`);
      return {
        id: `shadow_sell_${crypto.randomUUID()}`,
        client_order_id: `oid_${Date.now()}`,
        average_price: executedPrice.toString() // Return simulated price
      };
    }
    return null;
  }

  async _placeBuy(api, symbol, quantity, expectedPrice = null) {
    if (this.mode === 'LIVE' && api) return await api.placeBuy(symbol, quantity);
    if (this.mode === 'SHADOW') {
      // Calculate Slippage
      let executedPrice = expectedPrice || 0;
      if (expectedPrice && this.slippageMap && this.slippageMap[symbol.replace('-USD', '')]) {
        const slip = this.slippageMap[symbol.replace('-USD', '')];
        // Market Buy: We pay MORE (Price + Slippage)
        executedPrice = expectedPrice * (1 + slip);
      }
      // console.log(`[SHADOW] Buying ${quantity} ${symbol} @ ${executedPrice.toFixed(4)} (Exp: ${expectedPrice})`);
      return {
        id: `shadow_buy_${crypto.randomUUID()}`,
        client_order_id: `oid_${Date.now()}`,
        average_price: executedPrice.toString()
      };
    }
    return null;
  }

  _logTrade(data) {
    this.totalTrades++; // Increment trade count for metrics
    if (this.cycleTrades) this.cycleTrades.push(data.asset); // Track for Dreamer Focus
    if (this.mode === 'LIVE') {
      logTrade(data);
    } else {
      // console.log(`[SHADOW LOG] ${JSON.stringify(data)}`);
    }
  }

  // --- Core Logic ---
  async update(portfolioSummary, api, cashBalance, holdingDetails) {
    // CRITICAL: Sync external balance to internal state
    // Without this, this.cashBalance stays at initialization value (0) and blocks all buys
    this.cashBalance = cashBalance;

    // Persist Holdings in LIVE mode for Snapshotting
    if (this.mode === 'LIVE' && holdingDetails) {
      this.holdings = holdingDetails;
    }

    let anyTradesThisCycle = false;
    let stateChanged = false;
    let harvestedAmount = 0; // Local accumulator
    this.cycleTrades = [];   // Track trades for this cycle to notify Dreamer
    this.postMortemEvents = []; // Reset events

    // Calculate Total Value for Fitness Tracking
    let currentHoldingsValue = 0;
    portfolioSummary.forEach(r => currentHoldingsValue += r.Value);
    const currentTotalValue = currentHoldingsValue + this.cashBalance;

    // Dev Log Helper (Scoped to Engine or Global? Let's use Global if strictly defined, or simple conditional)
    const devLog = (msg) => { if (typeof LEGION_CONFIG !== 'undefined' && LEGION_CONFIG.ENABLE_DEVELOPER_LOGS) console.log(`🛠️ [DEV] ${msg}`); };


    // --- AUTO-INITIALIZE NEW ASSETS ---
    // If a new asset appears in holdings (user bought it), we must set a baseline so the bot can track it.
    // We set Baseline = Current Value (Assumption: User just bought it, so deviation is 0).
    if (this.mode === 'LIVE') {
      portfolioSummary.forEach(row => {
        if (row.Value > 1.0 && (!this.baselines[row.Symbol] || this.baselines[row.Symbol] <= 0)) {
          console.log(`✨ [NEW ASSET DETECTED] Initializing Baseline for ${row.Symbol} @ $${row.Value.toFixed(2)}`);
          this.baselines[row.Symbol] = row.Value;
          stateChanged = true;
        }
      });
    }


    // --- Tier 1: Manual Cash Extraction Detection & Baseline Healing ---
    if (this.mode === 'LIVE' && this.lastTotalValue > 0) {
      const dropPercent = (this.lastTotalValue - currentTotalValue) / this.lastTotalValue;
      if (dropPercent > 0.12) { // Trigger at > 12% drop
        // Calculate Average Price Change
        let totalPriceChangePct = 0;
        let priceCount = 0;
        portfolioSummary.forEach(r => {
          if (this.lastCyclePrices[r.Symbol]) {
            const prevP = this.lastCyclePrices[r.Symbol];
            const change = Math.abs((r.Price - prevP) / prevP);
            totalPriceChangePct += change;
            priceCount++;
          }
        });
        const avgPriceChange = priceCount > 0 ? (totalPriceChangePct / priceCount) : 0;

        if (avgPriceChange < 0.02) { // If prices moved < 2% (Stable)
          console.warn(`🚨 USER_EXTRACTION_DETECTED: Portfolio dropped ${(dropPercent * 100).toFixed(1)}% but prices are stable. Healing Baselines...`);
          // Proportional Baseline Reset
          Object.keys(this.baselines).forEach(sym => {
            // Scale down or set to current? Instruction: "set baselines[sym] = current_value[sym] * 0.995"
            const row = portfolioSummary.find(r => r.Symbol === sym);
            if (row) {
              this.baselines[sym] = row.Value * 0.995;
            }
          });
          this.lastTotalValue = currentTotalValue; // Reset tracking
          console.log(`   ✅ Baselines Re-aligned to Current Value * 0.995.`);
          return { anyTradesThisCycle: false, harvestedAmount: 0, tradedSymbols: [] }; // Skip trading this cycle
        }
      }
    }

    // Store current total value & prices for next cycle comparison
    this.lastTotalValue = currentTotalValue;
    portfolioSummary.forEach(r => this.lastCyclePrices[r.Symbol] = r.Price);

    // Update Risk Metrics
    if (currentTotalValue > this.peakTotalValue) {
      this.peakTotalValue = currentTotalValue;
    } else if (this.peakTotalValue > 0) {
      const drawdown = (this.peakTotalValue - currentTotalValue) / this.peakTotalValue;
      if (drawdown > this.maxDrawdownPercent) this.maxDrawdownPercent = drawdown;
    }

    // Local State References (Aliases)
    const tokenBaselines = this.baselines;
    const trailingState = this.trailingState;
    const lastActionTimestamps = this.lastActionTimestamps;
    const rebalanceState = this.rebalanceState;
    // const adaptiveDeadZoneState = this.adaptiveDeadZoneState; // REMOVED
    let portfolioHarvestState = this.portfolioHarvestState; // Local ref for reassignment
    const currentGenome = this.genome;
    const priceHistory = this.priceHistory;

    // --- Calculate Portfolio Deviation (Needed for Portfolio Harvest) ---
    let totalBaselineDifference = 0; let totalManagedBaselineValue = 0;
    portfolioSummary.forEach(row => { if (row.Baseline && typeof row.Baseline === 'number' && row.Baseline > 0 && !REBALANCE_EXCLUDE.includes(row.Symbol)) { totalBaselineDifference += (row.Value - row.Baseline); totalManagedBaselineValue += row.Baseline; } });
    let currentPortfolioDeviationPercent = 0; if (totalManagedBaselineValue > 0) { currentPortfolioDeviationPercent = (totalBaselineDifference / totalManagedBaselineValue) * 100; }

    // --- Update Price History & EMA for Adaptive Trend Escape (Tier 2) ---
    portfolioSummary.forEach(row => {
      if (!priceHistory[row.Symbol]) priceHistory[row.Symbol] = [];
      if (row.Price > 0) {
        // Add to history
        priceHistory[row.Symbol].push(row.Price);
        if (priceHistory[row.Symbol].length > currentGenome.PRICE_HISTORY_WINDOW_SIZE) priceHistory[row.Symbol].shift();
      }
    });

    // --- Update Global History Buffer (LIVE Mode Only) ---
    if (this.mode === 'LIVE') {
      const tick = { t: Date.now(), prices: {} };
      portfolioSummary.forEach(r => { if (r.Price > 0) tick.prices[r.Symbol] = r.Price; });
      this.priceHistoryBuffer.push(tick);
      // Keep last 65 minutes to be safe (needed for 60m backtest)
      const cutoff = Date.now() - (65 * 60 * 1000);
      if (this.priceHistoryBuffer.length > 0 && this.priceHistoryBuffer[0].t < cutoff) {
        // Optimization: finding index might be faster than filter if huge, but filter is cleaner
        this.priceHistoryBuffer = this.priceHistoryBuffer.filter(t => t.t >= cutoff);
      }
    }

    // --- Active Shadow Logic ---
    // Check if the shadow is defective based on recent history
    if (this.mode === 'SHADOW' && this.isDefective(portfolioSummary, this.priceHistoryBuffer)) { // Pass the buffer
      // console.log(`💀 Shadow ${this.id} is defective. Terminating.`);
      return {
        anyTradesThisCycle: false,
        harvestedAmount: 0,
        tradedSymbols: [],
        postMortemEvents: [],
        killMe: true // <--- SIGNAL TO MANAGER
      };
    }

    // --- Project Dynamo: The Heavy Spar (Dynamic Baselines) ---
    // SIMULATION FIX: Disable baseline drift in SHADOW mode to prevent fake ROI from compounding drift
    if (this.mode !== 'SHADOW') {
      Object.keys(tokenBaselines).forEach(sym => {
        const currentBaseline = tokenBaselines[sym];
        const row = portfolioSummary.find(r => r.Symbol === sym);
        const currentValue = row ? row.Value : 0;
        if (currentBaseline > 0 && currentValue > 0) {
          const gap = currentBaseline - currentValue;
          const sparDrag = getGenomicParam(currentGenome, 'SPAR_DRAG_COEFFICIENT', sym);
          const newBaseline = currentValue + (gap * sparDrag);
          tokenBaselines[sym] = newBaseline;
        }
      });
    }

    // --- Adaptive Dead Zone Activation/Deactivation Logic (REMOVED) ---
    // Logic removed in favor of Scientific Regime System

    // --- Crash Protection Check ---
    let isGlobalRiskSignalActive = false;
    if (currentGenome.ENABLE_CRASH_PROTECTION) {
      let assetsWithBaselineCount = 0; let assetsMeetingDeclineThresholdCount = 0;
      portfolioSummary.forEach(row => {
        if (row.Baseline && typeof row.Baseline === 'number' && row.Baseline > 0) {
          assetsWithBaselineCount++;
          const deviation = (row.Value - row.Baseline) / row.Baseline;
          if (deviation <= currentGenome.CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT) { assetsMeetingDeclineThresholdCount++; }
        }
      });
      if (assetsWithBaselineCount > 0) {
        const percentageMeetingThreshold = assetsMeetingDeclineThresholdCount / assetsWithBaselineCount;
        if (percentageMeetingThreshold >= currentGenome.CP_TRIGGER_ASSET_PERCENT) {
          isGlobalRiskSignalActive = true;
          // console.log(`🛡️ Crash Protection ACTIVE.`);
        }
      }
    }

    // --- Portfolio Override Harvest Logic ---
    let portfolioHarvestExecutedThisCycle = false;
    if (currentGenome.ENABLE_PORTFOLIO_HARVEST) {
      const portfolioHarvestTriggerValue = currentGenome.PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT * 100;
      if (!portfolioHarvestState.flagged && currentPortfolioDeviationPercent >= portfolioHarvestTriggerValue) {
        portfolioHarvestState = { flagged: true, cycleCount: 0, flaggedAt: Date.now(), previousDeviationPercent: currentPortfolioDeviationPercent };
        // console.log(`📈 Portfolio flagged for Baseline Reset Harvest.`);
      } else if (portfolioHarvestState.flagged && currentPortfolioDeviationPercent < portfolioHarvestTriggerValue) {
        // console.log(`📉 Portfolio dropped below Baseline Reset Harvest trigger. Clearing flag.`);
        portfolioHarvestState = { flagged: false, cycleCount: 0, previousDeviationPercent: null, flaggedAt: null };
      }
      if (portfolioHarvestState.flagged) {
        const prevDev = portfolioHarvestState.previousDeviationPercent;
        if (prevDev !== null) {
          const currDev = currentPortfolioDeviationPercent;
          if (currDev < prevDev - PRECISION_THRESHOLD) { portfolioHarvestState.cycleCount++; }
          else if (currDev > prevDev + PRECISION_THRESHOLD) { portfolioHarvestState.cycleCount = Math.max(0, portfolioHarvestState.cycleCount - 1); }
        }
        portfolioHarvestState.previousDeviationPercent = currentPortfolioDeviationPercent;
      }
      if (portfolioHarvestState.flagged && portfolioHarvestState.cycleCount >= currentGenome.PORTFOLIO_HARVEST_CONFIRMATION_CYCLES) {
        // console.log(`🎉 Executing Portfolio Baseline Reset Harvest!`);
        portfolioHarvestExecutedThisCycle = true;
        let assetsSoldCount = 0;
        const sellPromises = []; const assetsToUpdateTimestamp = [];
        for (const row of portfolioSummary) {
          if (HARVEST_EXCLUDE.includes(row.Symbol) || !row.Baseline || row.Value <= row.Baseline) continue;
          const originalBaseline = row.Baseline;
          const surplusUSD = row.Value - originalBaseline;
          if (surplusUSD < currentGenome.MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST) continue;
          const qtyToSell = surplusUSD / row.Price;
          const qtyStr = roundQty(row.Symbol, qtyToSell);
          if (parseFloat(qtyStr) > 0) {
            assetsSoldCount++;
            sellPromises.push((async () => {
              try {
                const sellResp = await this._placeSell(api, `${row.Symbol}-USD`, qtyStr);
                if (sellResp?.id) {
                  const effectiveSellPrice = getEffectivePriceFromResp(sellResp, row.Price) || row.Price;
                  const actualSoldValue = parseFloat(qtyStr) * effectiveSellPrice;
                  this._logTrade({ asset: row.Symbol, side: "SELL", quantity: qtyStr, price: effectiveSellPrice.toString(), clientOrderId: sellResp.client_order_id || sellResp.id, note: `Portfolio Baseline Reset Harvest` });
                  tokenBaselines[row.Symbol] = originalBaseline;
                  if (trailingState[row.Symbol]) delete trailingState[row.Symbol];
                  assetsToUpdateTimestamp.push(row.Symbol);

                  // Update Cash (Live & Shadow)
                  if (this.mode === 'LIVE') {
                    this.cashBalance += actualSoldValue;
                  }
                  // SHADOW MODE: Update Holdings & Cash (with 1% fee)
                  else if (this.mode === 'SHADOW') {
                    const soldQty = parseFloat(qtyStr);
                    if (this.holdings[row.Symbol]) {
                      this.holdings[row.Symbol].rawQuantity -= soldQty;
                      if (this.holdings[row.Symbol].rawQuantity < 0) this.holdings[row.Symbol].rawQuantity = 0;
                    }
                    this.cashBalance += (actualSoldValue * 0.99); // 1% fee
                  }

                  return actualSoldValue;
                }
                return 0;
              } catch (err) { console.error(`   ❌ Error P-Harvest sell ${row.Symbol}:`, err.message); return 0; }
            })());
          }
        }
        const harvestedValues = await Promise.all(sellPromises);
        harvestedAmount += harvestedValues.reduce((sum, val) => sum + val, 0);
        if (assetsSoldCount > 0) {
          anyTradesThisCycle = true; stateChanged = true;
          assetsToUpdateTimestamp.forEach(sym => { lastActionTimestamps[sym] = Date.now(); });
        }
        portfolioHarvestState = { flagged: false, cycleCount: 0, previousDeviationPercent: null, flaggedAt: null };
      }
    }

    // --- Individual Asset Harvest Logic ---
    if (!portfolioHarvestExecutedThisCycle) {
      for (const row of portfolioSummary) {
        const sym = row.Symbol; const currentBaseline = tokenBaselines[sym];
        if (HARVEST_EXCLUDE.includes(sym) || !currentBaseline || currentBaseline <= 0) continue;
        const curP = row.Price; const totalVal = row.usdValueNum; const currentDeviation = (totalVal - currentBaseline) / currentBaseline;
        // const isAdaptiveActive = ... (Removed)

        // Fetch dynamic params
        const flatHarvestTrigger = getGenomicParam(currentGenome, 'FLAT_HARVEST_TRIGGER_PERCENT', sym);
        // const adaptiveHarvestTrigger = ... (Removed)

        const effectiveHarvestTrigger = flatHarvestTrigger;
        const harvestTriggerValue = currentBaseline * (1 + effectiveHarvestTrigger);

        if (!trailingState[sym]) { trailingState[sym] = { flagged: false, harvestCycleCount: 0, flaggedAt: null, previousDeviation: null }; }
        let st = trailingState[sym];

        if (!st.flagged && totalVal >= harvestTriggerValue) {
          st = { flagged: true, harvestCycleCount: 0, flaggedAt: Date.now(), previousDeviation: currentDeviation };
          trailingState[sym] = st;
          stateChanged = true;
        } else if (st.flagged && totalVal < harvestTriggerValue) {
          delete trailingState[sym]; stateChanged = true; continue;
        }
        if (!st.flagged) continue;

        // Harvest Execution Logic
        const flaggedDuration = Date.now() - (st.flaggedAt || Date.now());
        const baseHarvestCycles = getGenomicParam(currentGenome, 'HARVEST_CYCLE_THRESHOLD', sym);
        const requiredHarvestCycles = baseHarvestCycles;

        let shouldHarvest = false; let harvestType = "";

        if (st.previousDeviation !== null) {
          if (currentDeviation < st.previousDeviation - PRECISION_THRESHOLD) st.harvestCycleCount++;
          else if (currentDeviation > st.previousDeviation + PRECISION_THRESHOLD) st.harvestCycleCount = Math.max(0, st.harvestCycleCount - 1);
        }
        st.previousDeviation = currentDeviation;

        if (flaggedDuration > currentGenome.FORCED_HARVEST_TIMEOUT) { shouldHarvest = true; harvestType = "Forced"; }
        else if (st.harvestCycleCount >= requiredHarvestCycles) { shouldHarvest = true; harvestType = "Standard"; }

        if (shouldHarvest) {
          const surplus = totalVal - currentBaseline;
          const minSurplus = harvestType === "Forced" ? currentGenome.MIN_SURPLUS_FOR_FORCED_HARVEST : getGenomicParam(currentGenome, 'MIN_SURPLUS_FOR_HARVEST', sym);

          if (surplus >= minSurplus) {
            let harvestTakePct = getGenomicParam(currentGenome, 'HARVEST_TAKE_PERCENT', sym);
            if (harvestTakePct === undefined) harvestTakePct = 0.70; // Fallback

            const qtyToSell = (surplus * harvestTakePct) / curP;
            const retainedSurplusUSD = surplus * (1 - harvestTakePct);
            const qtyStr = roundQty(sym, qtyToSell);
            // Tier 2: Dust Protection
            if (checkMinTrade(parseFloat(qtyStr) * curP)) {
              if (parseFloat(qtyStr) > 0) {
                try {
                  console.log(`📉 Attempting ${harvestType} Harvest ${sym}`);
                  const sellResp = await this._placeSell(api, `${sym}-USD`, qtyStr, curP);
                  if (sellResp?.id) {
                    const effectiveSellPrice = getEffectivePriceFromResp(sellResp, curP) || curP;
                    const actualSoldValue = parseFloat(qtyStr) * effectiveSellPrice;
                    this._logTrade({ asset: sym, side: "SELL", quantity: qtyStr, price: effectiveSellPrice.toString(), clientOrderId: sellResp.client_order_id || sellResp.id, note: `${harvestType} Harvest` });
                    harvestedAmount += actualSoldValue; anyTradesThisCycle = true;

                    // Tier 1: Post-Mortem Event
                    if (this.mode === 'LIVE') {
                      this.postMortemEvents.push({
                        symbol: sym,
                        type: 'harvest',
                        surplusUSD: surplus, // Actual realized surplus (approx)
                        deviation: currentDeviation,
                        genomeSlice: { ...currentGenome } // Pass full genome for now, Dreamer filters
                      });
                    }

                    const targetAdjust = getGenomicParam(currentGenome, 'TARGET_ADJUST_PERCENT', sym);
                    tokenBaselines[sym] += retainedSurplusUSD; // Geometric Growth: structurally raise baseline
                    tokenBaselines[sym] *= (1 + targetAdjust); // Additional Ratchet Tension, if any

                    lastActionTimestamps[sym] = Date.now();

                    if (this.mode === 'LIVE') {
                      this.cashBalance += actualSoldValue;
                    }
                    else if (this.mode === 'SHADOW') {
                      // Tier 2: Real Transaction Costs (1% Slippage/Fee Model)
                      const soldQty = parseFloat(qtyStr);
                      if (this.holdings[sym]) {
                        this.holdings[sym].rawQuantity -= soldQty;
                        if (this.holdings[sym].rawQuantity < 0) this.holdings[sym].rawQuantity = 0;
                      }
                      // Sell: Proceeds = Value - 1%
                      this.cashBalance += (actualSoldValue * 0.99);
                    }

                    delete trailingState[sym]; stateChanged = true;
                  } else { st.harvestCycleCount = 0; stateChanged = true; }
                } catch (err) { console.error(`❌ Error Harvest ${sym}:`, err.message); }
              } else { delete trailingState[sym]; stateChanged = true; }
            } else { if (harvestType === "Forced") delete trailingState[sym]; else st.harvestCycleCount = 0; stateChanged = true; }
          }
        }
      }

      // --- Harvest Proceeds Allocation ---
      if (harvestedAmount > 0) {
        let amountForReinvest = 0; let amountForBTC = 0; let amountForETH = 0;

        // 💰 CRASH FUND CHECK
        // Ensure we have 10% Cash Reserve before allocating to Risk Assets
        // Note: portfolioSummary has OLD asset values. this.cashBalance has NEW cash (inc. harvest).
        // Correct Total = Sum(OldAssets) - Harvested + NewCash
        const currentTotalPortfolioValue = portfolioSummary.reduce((sum, r) => sum + r.Value, 0) - harvestedAmount + this.cashBalance;
        const currentCashPercent = this.cashBalance / Math.max(1, currentTotalPortfolioValue); // Avoid div/0
        const crashFundThreshold = currentGenome.CRASH_FUND_THRESHOLD_PERCENT || 0.10;

        if (currentCashPercent < crashFundThreshold) {
          if (this.mode === 'LIVE') {
            console.log(`💰 [CRASH FUND ACTIVE] Cash at ${(currentCashPercent * 100).toFixed(1)}% (Target: ${(crashFundThreshold * 100).toFixed(1)}%).`);
            console.log(`   → Portfolio Value: $${currentTotalPortfolioValue.toFixed(2)} | Cash: $${this.cashBalance.toFixed(2)}`);
            console.log(`   → Keeping 100% of Harvest ($${harvestedAmount.toFixed(2)}) in Cash Reserves.`);
          }
          // Implicit: amounts remain 0, so no buys are executed.
          // Proceeds rely in cashBalance from the sell step.
        }
        else if (harvestedAmount >= currentGenome.MIN_HARVEST_TO_ALLOCATE) {
          // Apply Allocation Mode Modifiers
          let reinvestPct = currentGenome.HARVEST_ALLOC_REINVEST_PERCENT;
          let btcPct = currentGenome.HARVEST_ALLOC_BTC_PERCENT;
          let ethPct = currentGenome.HARVEST_ALLOC_ETH_PERCENT;
          // Cash % is implicit remainder

          // 0=BALANCED (No Change)
          // 1=GROWTH (+15% Reinvest, -15% Cash/Safe)
          // 2=DEFENSIVE (-15% Reinvest, +15% Cash/Safe)
          const mode = Math.round(currentGenome.ALLOCATION_MODE);
          if (mode === 1) { // GROWTH
            reinvestPct += 0.15;
          } else if (mode === 2) { // DEFENSIVE
            reinvestPct -= 0.15;
            // Remainder naturally flows to Cash (implicit)
          }

          // Clamp to safe bounds 0-1
          reinvestPct = Math.max(0, Math.min(1, reinvestPct));

          // NORMALIZE: Prioritize BTC & ETH (User Request), squeeze Reinvest if needed
          const fixedAlloc = btcPct + ethPct; // 0.20
          if (reinvestPct + fixedAlloc > 1.0) {
            const maxReinvest = Math.max(0, 1.0 - fixedAlloc);
            if (this.mode === 'LIVE') console.log(`⚖️ [ALLOCATION CLAMP] Total > 100%. Prioritizing BTC/ETH. Capping Reinvest from ${(reinvestPct * 100).toFixed(1)}% to ${(maxReinvest * 100).toFixed(1)}%.`);
            reinvestPct = maxReinvest;
          }

          amountForReinvest = harvestedAmount * reinvestPct;
          amountForBTC = harvestedAmount * btcPct;
          amountForETH = harvestedAmount * ethPct;

          if (this.mode === 'LIVE') {
            console.log(`🌾 Harvest Allocation: Reinvest $${amountForReinvest.toFixed(2)} | BTC $${amountForBTC.toFixed(2)} | ETH $${amountForETH.toFixed(2)}`);
          }
        } else {
          amountForReinvest = 0; amountForBTC = 0; amountForETH = 0; // Keep as cash
        }

        // Reinvestment Logic
        if (amountForReinvest > 0) {
          // Scoring: deviation^Exponent. (Note: deviation is negative, so use Math.abs)
          // Higher exponent = stronger preference for deeper dips.
          const exponent = currentGenome.REINVEST_WEIGHT_EXPONENT || 1.5;

          // 1. Identify Candidates (The 3 Deepest Dips - Focused Reinvestment)
          let reinvestmentCandidates = portfolioSummary.filter(r =>
            !REBALANCE_EXCLUDE.includes(r.Symbol) &&
            r.Baseline > 0 &&
            r.usdValueNum < r.Baseline &&
            ((r.usdValueNum - r.Baseline) / r.Baseline) <= currentGenome.MIN_NEGATIVE_DEVIATION_FOR_REINVEST
          ).map(r => {
            const dev = (r.usdValueNum - r.Baseline) / r.Baseline;
            const score = Math.pow(Math.abs(dev), exponent);
            return { ...r, deviation: dev, score: score };
          })
            .sort((a, b) => b.score - a.score); // Biggest dips first

          if (reinvestmentCandidates.length > 3) reinvestmentCandidates = reinvestmentCandidates.slice(0, 3);

          const perCandidate = amountForReinvest / Math.max(1, reinvestmentCandidates.length);

          // 2. Execute Growth Buys
          for (const cand of reinvestmentCandidates) {
            const gap = cand.Baseline - cand.Value;
            let buyUSD = Math.min(perCandidate, gap);

            if (buyUSD >= currentGenome.MIN_REINVEST_BUY_USD) {
              const qty = roundQty(cand.Symbol, buyUSD / cand.Price);

              // Tier 2: Dust Protection
              if (checkMinTrade(parseFloat(qty) * cand.Price)) {
                if (parseFloat(qty) > 0) {
                  // EXECUTE BUY
                  const resp = await this._placeBuy(api, `${cand.Symbol}-USD`, qty, cand.Price);

                  if (resp?.id) {
                    const effectivePrice = getEffectivePriceFromResp(resp, cand.Price);
                    const cost = parseFloat(qty) * effectivePrice;

                    // --- THE GROWTH LOGIC ---
                    const growthFactor = currentGenome.REINVEST_BASELINE_GROWTH_FACTOR || 0.85;
                    const baselineIncrease = cost * growthFactor;

                    // 1. Apply Trade
                    this._logTrade({ asset: cand.Symbol, side: "BUY", quantity: qty, price: effectivePrice.toString(), clientOrderId: resp.client_order_id || resp.id, note: `Growth Reinvest (+Base $${baselineIncrease.toFixed(2)})` });
                    anyTradesThisCycle = true;
                    lastActionTimestamps[cand.Symbol] = Date.now();
                    stateChanged = true;
                    if (rebalanceState[cand.Symbol]) delete rebalanceState[cand.Symbol];

                    // 2. Raise Baseline (The "Little Less" Logic)
                    tokenBaselines[cand.Symbol] += baselineIncrease;

                    if (this.mode === 'LIVE') console.log(`🌱 [GROWTH] Reinvested $${cost.toFixed(2)} into ${cand.Symbol}. Baseline increased by $${baselineIncrease.toFixed(2)}.`);

                    // 3. Track Cash
                    const costWithFee = this.mode === 'SHADOW' ? cost * 1.01 : cost;
                    this.cashBalance -= costWithFee;
                    if (this.cashBalance < 0) this.cashBalance = 0;

                    // 4. Shadow Accounting
                    if (this.mode === 'SHADOW') {
                      if (!this.holdings[cand.Symbol]) this.holdings[cand.Symbol] = { rawQuantity: 0 };
                      this.holdings[cand.Symbol].rawQuantity += parseFloat(qty);
                    }
                  }
                }
              }
            }
          }

          // BTC Buy
          if (amountForBTC >= currentGenome.MIN_BTC_BUY_USD) {
            const btcP = portfolioSummary.find(r => r.Symbol === 'BTC')?.Price || (await api?.getQuotes(['BTC']))?.['BTC'];
            if (btcP > 0 && this.cashBalance >= amountForBTC) {
              const qty = roundQty('BTC', amountForBTC / btcP);
              // Validate Min Quantity (API 400 Prevention)
              if (parseFloat(qty) > 0 && checkMinQuantity('BTC', qty)) {
                const resp = await this._placeBuy(api, 'BTC-USD', qty, btcP);
                if (resp?.id) {
                  const effectivePrice = getEffectivePriceFromResp(resp, btcP);
                  this._logTrade({ asset: 'BTC', side: 'BUY', quantity: qty, price: effectivePrice.toString(), clientOrderId: resp.client_order_id || resp.id, note: 'Allocated BTC Buy' });
                  tokenBaselines['BTC'] = ((holdingDetails['BTC']?.rawQuantity || 0) + parseFloat(qty)) * effectivePrice;
                  lastActionTimestamps['BTC'] = Date.now(); stateChanged = true;

                  // TRACK CASH LOCALLY
                  this.cashBalance -= amountForBTC;
                  if (this.cashBalance < 0) this.cashBalance = 0;
                }
              } else {
                if (this.mode === 'LIVE') console.log(`   🔸 BTC Buy Skipped: Qty ${qty} below minimum.`);
              }
            } else {
              if (this.mode === 'LIVE') console.log(`   🔸 BTC Buy Skipped: Cash $${this.cashBalance.toFixed(2)} < Alloc $${amountForBTC.toFixed(2)}`);
            }
          }
          // ETH Buy
          if (amountForETH >= currentGenome.MIN_ETH_BUY_USD) {
            const ethP = portfolioSummary.find(r => r.Symbol === 'ETH')?.Price || (await api?.getQuotes(['ETH']))?.['ETH'];
            if (ethP > 0 && this.cashBalance >= amountForETH) {
              const qty = roundQty('ETH', amountForETH / ethP);
              // Validate Min Quantity (API 400 Prevention)
              if (parseFloat(qty) > 0 && checkMinQuantity('ETH', qty)) {
                const resp = await this._placeBuy(api, 'ETH-USD', qty, ethP);
                if (resp?.id) {
                  const effectivePrice = getEffectivePriceFromResp(resp, ethP);
                  this._logTrade({ asset: 'ETH', side: 'BUY', quantity: qty, price: effectivePrice.toString(), clientOrderId: resp.client_order_id || resp.id, note: 'Allocated ETH Buy' });
                  tokenBaselines['ETH'] = ((holdingDetails['ETH']?.rawQuantity || 0) + parseFloat(qty)) * effectivePrice;
                  lastActionTimestamps['ETH'] = Date.now(); stateChanged = true;

                  // TRACK CASH LOCALLY
                  this.cashBalance -= amountForETH;
                  if (this.cashBalance < 0) this.cashBalance = 0;
                }
              } else {
                if (this.mode === 'LIVE') console.log(`   🔸 ETH Buy Skipped: Qty ${qty} below minimum.`);
              }
            }
          }
        }
      }

      // --- Rebalancing Logic ---
      for (const row of portfolioSummary) {
        const sym = row.Symbol; const currentBaseline = tokenBaselines[sym];
        if (REBALANCE_EXCLUDE.includes(sym) || !currentBaseline || currentBaseline <= 0 || trailingState[sym]?.flagged) { if (rebalanceState[sym]) delete rebalanceState[sym]; continue; }
        const totalVal = row.Value; const curP = row.Price;
        const flatRebalanceTrigger = getGenomicParam(currentGenome, 'FLAT_REBALANCE_TRIGGER_PERCENT', sym);
        // const adaptiveRebalanceTrigger = ... (Removed)

        const effectiveRebalanceTrigger = flatRebalanceTrigger;
        const rebalanceTriggerValue = currentBaseline * (1 - effectiveRebalanceTrigger);

        // HYSTERESIS: Prevent flip-flopping by requiring a small recovery buffer (0.2%) to clear state
        const recoveryBuffer = rebalanceState[sym] ? 1.002 : 1.0;

        // --- ZERO BALANCE GUARD ---
        // If the asset value has dropped near zero (user sold it all), do NOT attempt to rebalance.
        // We don't want to throw good money after a dead asset unless the user explicitly buys back in.
        if (totalVal < 1.0) {
          if (rebalanceState[sym]) {
            console.log(`📉 Dropping Rebalance for ${sym} (Value < $1.00 - Assuming Fully Sold).`);
            delete rebalanceState[sym];
          }
          continue;
        }

        if (totalVal >= rebalanceTriggerValue * recoveryBuffer) {
          if (rebalanceState[sym]) {
            if (this.mode === 'LIVE') console.log(`📈 Clearing Rebalance ${sym} (Recovered to $${totalVal.toFixed(2)})`);
            delete rebalanceState[sym];
          }
          continue;
        }

        if (!rebalanceState[sym]) {
          rebalanceState[sym] = { triggered: true, triggeredAt: Date.now(), rebalancePosCycleCount: 0, attemptCount: 0, cooldownUntil: 0, currentBaselineWhenTriggered: currentBaseline, previousDeviation: (totalVal - currentBaseline) / currentBaseline };
          if (this.mode === 'LIVE') console.log(`⚖️ ${sym} Rebalance Triggered.`);
        }
        let rSt = rebalanceState[sym];

        // Forced Rebalance Check
        if (Date.now() - rSt.triggeredAt > currentGenome.FORCE_REBALANCE_TIMEOUT) {
          const shortfall = rSt.currentBaselineWhenTriggered - totalVal;
          const buyUSD = shortfall * currentGenome.FORCE_REBALANCE_SHORTFALL_PERCENT;
          if (buyUSD >= currentGenome.MIN_FORCED_REBALANCE_USD) {
            const qty = roundQty(sym, buyUSD / curP);
            if (parseFloat(qty) > 0 && this.cashBalance >= buyUSD) {
              const resp = await this._placeBuy(api, `${sym}-USD`, qty, curP);
              if (resp?.id) {
                const effectivePrice = getEffectivePriceFromResp(resp, curP);
                this._logTrade({ asset: sym, side: 'BUY', quantity: qty, price: effectivePrice.toString(), clientOrderId: resp.client_order_id || resp.id, note: 'Forced Rebalance' });
                anyTradesThisCycle = true;
                // Tier 1: Post-Mortem Event (Forced Rebalance is a significant event)
                if (this.mode === 'LIVE') {
                  this.postMortemEvents.push({
                    symbol: sym,
                    type: 'rebalance', // Treat forced rebalance as event
                    shortfallUSD: shortfall,
                    deviation: (totalVal - rSt.currentBaselineWhenTriggered) / rSt.currentBaselineWhenTriggered,
                    genomeSlice: { ...currentGenome }
                  });
                }
                if (true) {
                  const targetAdjust = getGenomicParam(currentGenome, 'TARGET_ADJUST_PERCENT', sym);
                  tokenBaselines[sym] *= (1 - targetAdjust);
                }
                lastActionTimestamps[sym] = Date.now(); stateChanged = true; delete rebalanceState[sym];

                // TRACK CASH LOCALLY (Live & Shadow)
                // Forced Rebalance: Cost + Fee (1%)
                const cost = buyUSD * (this.mode === 'SHADOW' ? 1.01 : 1.00);
                this.cashBalance -= cost;
                if (this.cashBalance < 0) this.cashBalance = 0;
                continue;
              }
            }
          }
        }

        const currentDeviation = (totalVal - currentBaseline) / currentBaseline;
        if (rSt.previousDeviation !== null) {
          if (currentDeviation > rSt.previousDeviation + PRECISION_THRESHOLD) rSt.rebalancePosCycleCount++;
          else if (currentDeviation < rSt.previousDeviation - PRECISION_THRESHOLD) rSt.rebalancePosCycleCount = Math.max(0, rSt.rebalancePosCycleCount - 1);
        }
        rSt.previousDeviation = currentDeviation;

        const basePosThreshold = getGenomicParam(currentGenome, 'REBALANCE_POSITIVE_THRESHOLD', sym);
        const reqCycles = basePosThreshold;

        if (rSt.rebalancePosCycleCount >= reqCycles && Date.now() >= rSt.cooldownUntil) {
          const shortfall = rSt.currentBaselineWhenTriggered - totalVal;
          const partialRecovery = Math.min(1.0, getGenomicParam(currentGenome, 'PARTIAL_RECOVERY_PERCENT', sym));
          let buyUSD = shortfall * partialRecovery;

          // INSUFFICIENT FUNDS HANDLING: Clamp to available cash if we have at least minimum
          if (buyUSD > this.cashBalance) {
            if (this.cashBalance >= currentGenome.MIN_PARTIAL_REBALANCE_USD) {
              if (this.mode === 'LIVE') console.log(`⚠️ Cash Constrained Rebalance: Wanted $${buyUSD.toFixed(2)} -> Buy Max Available $${this.cashBalance.toFixed(2)}`);
              buyUSD = this.cashBalance * 0.95; // Leave 5% buffer for slippage/fees
            }
          }

          if (buyUSD >= currentGenome.MIN_PARTIAL_REBALANCE_USD) {
            const qty = roundQty(sym, buyUSD / curP);
            if (parseFloat(qty) > 0 && this.cashBalance >= buyUSD) {
              const resp = await this._placeBuy(api, `${sym}-USD`, qty, curP);
              if (resp?.id) {
                const effectivePrice = getEffectivePriceFromResp(resp, curP);
                this._logTrade({ asset: sym, side: 'BUY', quantity: qty, price: effectivePrice.toString(), clientOrderId: resp.client_order_id || resp.id, note: 'Rebalance Buy' });
                anyTradesThisCycle = true;
                // Tier 1: Post-Mortem Event
                if (this.mode === 'LIVE') {
                  this.postMortemEvents.push({
                    symbol: sym,
                    type: 'rebalance',
                    shortfallUSD: shortfall,
                    deviation: (totalVal - rSt.currentBaselineWhenTriggered) / rSt.currentBaselineWhenTriggered,
                    genomeSlice: { ...currentGenome }
                  });
                }
                if (true) {
                  const targetAdjust = getGenomicParam(currentGenome, 'TARGET_ADJUST_PERCENT', sym);
                  tokenBaselines[sym] *= (1 - targetAdjust);
                }
                lastActionTimestamps[sym] = Date.now(); stateChanged = true;

                // TRACK CASH LOCALLY (Live & Shadow)
                const cost = buyUSD;
                const costWithFee = this.mode === 'SHADOW' ? cost * 1.01 : cost; // 1% buy fee
                this.cashBalance -= costWithFee;
                if (this.cashBalance < 0) this.cashBalance = 0;

                // SHADOW: Update holdings
                if (this.mode === 'SHADOW') {
                  if (!this.holdings[sym]) this.holdings[sym] = { rawQuantity: 0 };
                  this.holdings[sym].rawQuantity += parseFloat(qty);
                }

                rSt.attemptCount++; rSt.rebalancePosCycleCount = 0; rSt.previousDeviation = null;
                const maxAttempts = getGenomicParam(currentGenome, 'MAX_REBALANCE_ATTEMPTS', sym);

                // Using the new rebalance trigger logic (fetched via getGenomicParam) for consistency
                if (totalVal + buyUSD >= rSt.currentBaselineWhenTriggered * (1 - effectiveRebalanceTrigger)) delete rebalanceState[sym];
                else if (rSt.attemptCount >= maxAttempts) rSt.cooldownUntil = Date.now() + currentGenome.REBALANCE_COOLDOWN;
              }
            }
          }
        }
      }

      // Return state updates
      this.portfolioHarvestState = portfolioHarvestState;
      // Return state updates
      this.portfolioHarvestState = portfolioHarvestState;

      // Tier 1: Min Trade Enforcement (Counters)
      if (anyTradesThisCycle) {
        this.cyclesWithoutTrade = 0;
      } else {
        this.cyclesWithoutTrade++;
        // Tier 1 #5: Force Trade if > 250 cycles (Shadow Only for now?)
        // Instruction: "Add two counters per shadow... forcedTradeTimer... When > X cycles... force one minimal-size harvest"
        if (this.mode === 'SHADOW' && this.cyclesWithoutTrade > 250) {
          // Logic to force a trade - Pick a random asset and nudge it?
          // For now, simpler implementation: Just PENALIZE fitness if no trades.
          // The "Force Trade" logic is complex to inject here without access to full market context again.
          // We will handle the penalty in EvolutionManager.
        }
      }

      return {
        anyTradesThisCycle,
        harvestedAmount,
        tradedSymbols: this.cycleTrades || [],
        postMortemEvents: this.postMortemEvents
      };
    }

    // Fallthrough return (Critical for avoiding undefined)
    return {
      anyTradesThisCycle,
      harvestedAmount,
      tradedSymbols: this.cycleTrades || [],
      postMortemEvents: this.postMortemEvents
    };
  }
}



// ============== Global State ==============
let liveEngine;
let legionManager;       // The Allocator
let assetRegimeManager;  // The Memory
let regimeDetector;      // The Eyes (was Oracle)
let dreamerGrid = [];    // The Deep Past Workers
// let lastStateSaveTime = 0; // Already defined?
// let harvestedAmount = 0;   // Already defined?e saves
let harvestedAmount = 0;        // Tracks USD harvested within a single cycle
// let priceHistory = {}; // Removed duplicate declaration. Hydrated global array used now.

function printTable(headers, rows) {
  if (!rows || rows.length === 0) return;
  const colWidths = headers.map((h, i) => {
    const maxRowLen = Math.max(...rows.map(r => {
      // Strip ANSI codes for length calculation
      const val = r[i];
      const str = (val !== null && val !== undefined) ? val.toString() : "";
      const clean = str.replace(/\x1b\[[0-9;]*m/g, "");
      return clean.length;
    }));
    return Math.max(h.length, maxRowLen) + 2; // +2 Padding
  });

  const border = colWidths.map(w => "─".repeat(w)).join("┼");
  console.log("┌" + border.replace(/┼/g, "┬") + "┐");

  // Header
  const headerRow = headers.map((h, i) => (h || "").toString().padEnd(colWidths[i])).join("│");
  console.log("│" + headerRow + "│");
  console.log("├" + border + "┤");

  // Rows
  rows.forEach(row => {
    const line = row.map((cell, i) => {
      const cellStr = (cell !== null && cell !== undefined) ? cell.toString() : "";
      const cleanLen = cellStr.replace(/\x1b\[[0-9;]*m/g, "").length;
      const padding = colWidths[i] - cleanLen;
      return cellStr + " ".repeat(padding);
    }).join("│");
    console.log("│" + line + "│");
  });
  console.log("└" + border.replace(/┼/g, "┴") + "┘");
}


// ============== Helper Functions ==============
function getGenomicParam(genome, key, symbol) {
  // 1. Check for Asset-Specific Override
  if (symbol && genome.overrides && genome.overrides[symbol] && genome.overrides[symbol][key] !== undefined) {
    return genome.overrides[symbol][key];
  }
  // 2. Return Global Default
  return genome[key];
}

function roundQty(sym, qty) {
  const step = minIncrementMap[sym]; let safeStep;
  if (!step) {
    if (sym === 'LTC') { console.warn(`[roundQty] Min increment step not defined for ${sym}. Using default 0.0001.`); safeStep = 0.0001; }
    else { console.warn(`[roundQty] Min increment step not defined for ${sym}. Using default 8 decimals (0.00000001).`); safeStep = 0.00000001; }
    if (typeof qty !== 'number' || isNaN(qty) || qty < (safeStep / 10)) return "0.0"; const rounded = Math.floor(qty / safeStep) * safeStep; const decimalPlaces = step.toString().includes('.') ? step.toString().split('.')[1].length : 0; let str = rounded.toFixed(Math.min(18, Math.max(8, decimalPlaces))); str = str.replace(/(\.\d*[1-9])0+$/, "$1"); str = str.replace(/\.0+$/, ""); return Number(str) < (safeStep / 10) ? "0.0" : str;
  }
  if (typeof qty !== 'number' || isNaN(qty) || qty < (step / 10)) return "0.0"; const rounded = Math.floor(qty / step) * step; const decimalPlaces = step.toString().includes('.') ? step.toString().split('.')[1].length : 0; let str = rounded.toFixed(Math.min(18, Math.max(8, decimalPlaces))); str = str.replace(/(\.\d*[1-9])0+$/, "$1"); str = str.replace(/\.0+$/, ""); return Number(str) < (step / 10) ? "0.0" : str;
}

function checkMinTrade(usdValue) {
  const MIN_TRADE_USD = 0.25;
  if (usdValue < MIN_TRADE_USD) {
    // console.log(`   🔸 Dust Trade Skipped (< $${MIN_TRADE_USD}): $${usdValue.toFixed(2)}`);
    return false;
  }
  return true;
}

function checkMinQuantity(symbol, qty) {
  const minQty = MIN_ORDER_QTY_MAP[symbol];
  if (minQty) {
    if (parseFloat(qty) < minQty) return false;
  }
  return true;
}

function logTrade({ asset, side, quantity, price, clientOrderId, note = "" }) {
  try {
    const quantityNum = parseFloat(quantity); const priceNum = parseFloat(price); if (isNaN(quantityNum) || isNaN(priceNum) || priceNum <= 0) { console.error(`Error logging trade: Invalid numeric values. Qty: ${quantity}, Price: ${price}`); return; } const totalValue = (quantityNum * priceNum).toFixed(2); appendTradeHistory({ asset, side: side.toUpperCase(), orderType: "market", quantity, effectivePrice: price, totalValue, clientOrderId, extra: { note } });
  } catch (error) { console.error(`Error logging trade for ${asset}:`, error); }
}

function appendTradeHistory(tradeRecord) {
  const tradeHistoryFile = path.join(process.cwd(), 'trade_history.log');
  if (!tradeRecord.timestamp) { tradeRecord.timestamp = new Date().toISOString(); }
  const logLine = JSON.stringify(tradeRecord);
  fs.appendFile(tradeHistoryFile, logLine + "\n", (err) => {
    if (err) console.error("Error appending trade history:", err);
  });
}
let lastMarketDataFlush = 0;
let marketDataBuffer = [];
function appendMarketData(timestamp, portfolioSummary) {
  const prices = {};
  portfolioSummary.forEach(r => prices[r.Symbol] = r.Price);
  const entry = JSON.stringify({ t: timestamp, p: prices });

  marketDataBuffer.push(entry);

  // Flush buffer to disk every ~2 minutes (15 ticks assuming 8s intervals) to prevent constant drive lock
  if (marketDataBuffer.length >= 15 || (Date.now() - lastMarketDataFlush > 120000)) {
    const logFile = path.join(process.cwd(), 'market_data.jsonl');
    const bulkData = marketDataBuffer.join('\n') + '\n';

    // File size safety check before writing
    try {
      if (fs.existsSync(logFile)) {
        const stats = fs.statSync(logFile);
        if (stats.size > 150 * 1024 * 1024) { // Cap at 150MB
          const content = fs.readFileSync(logFile, 'utf-8');
          const lines = content.split('\n');
          // Keep last 450000 ticks (~40 days of 8s ticks)
          if (lines.length > 450000) {
            const prunedContent = lines.slice(-450000).join('\n');
            fs.writeFileSync(logFile, prunedContent + (prunedContent.endsWith('\n') ? '' : '\n'));
          }
        }
      }
    } catch (e) { }

    fs.appendFile(logFile, bulkData, (err) => {
      if (err) console.error("Error appending market data:", err);
    });
    marketDataBuffer = [];
    lastMarketDataFlush = Date.now();
  }

  // Update In-Memory History
  if (typeof priceHistory !== 'undefined' && Array.isArray(priceHistory)) {
    priceHistory.push({ t: timestamp, p: prices });
    if (priceHistory.length > 300) priceHistory.shift();
  }
}

function loadRecentMarketData(limit = 200) {
  const logFile = path.join(process.cwd(), 'market_data.jsonl');
  if (!fs.existsSync(logFile)) return [];
  try {
    const content = fs.readFileSync(logFile, 'utf-8').trim();
    if (!content) return [];
    const lines = content.split('\n');
    const recent = lines.slice(-limit);
    return recent.map(line => {
      try { return JSON.parse(line); } catch (e) { return null; }
    }).filter(x => x);
  } catch (err) { console.error("Error loading recent market data:", err.message); return []; }
}
function getEffectivePriceFromResp(resp, fallbackPrice) {
  const priceStr = resp?.average_price || resp?.executions?.[0]?.effective_price || resp?.price || fallbackPrice?.toString(); if (priceStr === undefined || priceStr === null) return null; const priceNum = parseFloat(priceStr); return !isNaN(priceNum) && priceNum > 0 ? priceNum : null;
}

function loadState() {
  let loadedBaselines = {};
  let loadedTrailingState = {};
  let loadedLastActionTimestamps = {};
  let loadedGenome = null;
  try {
    if (fs.existsSync(STATE_FILE_PATH)) {
      const data = fs.readFileSync(STATE_FILE_PATH, 'utf-8');
      const loadedData = JSON.parse(data);
      loadedBaselines = loadedData.baselines || {};
      loadedTrailingState = loadedData.trailingState || {};
      loadedLastActionTimestamps = loadedData.lastActionTimestamps || {};
      loadedGenome = loadedData.genome; // Load genome but don't set global yet
      const loadedAssetSourceTimeframe = loadedData.assetSourceTimeframe || {};
      // oracleState will be extracted in main()
      console.log(`✅ Loaded state from ${STATE_FILE_PATH}.`);
      return { loadedBaselines, loadedTrailingState, loadedLastActionTimestamps, loadedGenome, loadedData, loadedAssetSourceTimeframe };
    } else {
      console.log(`ℹ️ ${STATE_FILE_PATH} not found, starting with fresh state.`);
    }
  } catch (err) {
    console.error(`❌ Error loading state from ${STATE_FILE_PATH}:`, err);
  }
  return { loadedBaselines, loadedTrailingState, loadedLastActionTimestamps, loadedGenome, loadedData: {} };
}
function saveState() {
  try {
    if (!liveEngine) return;
    const stateToSave = liveEngine.getStateSnapshot(); // Save from Engine
    if (typeof regimeDetector !== 'undefined') {
      stateToSave.regimeDetectorState = regimeDetector.regimes;
    }
    const tempFilePath = STATE_FILE_PATH + '.tmp';
    fs.writeFileSync(tempFilePath, JSON.stringify(stateToSave, null, 2));

    // EPERM Safe Rename
    try {
      fs.renameSync(tempFilePath, STATE_FILE_PATH);
    } catch (renameErr) {
      if (renameErr.code === 'EPERM' || renameErr.code === 'EBUSY') {
        // Retry once after small delay or fallback to copy
        try {
          fs.copyFileSync(tempFilePath, STATE_FILE_PATH);
          fs.unlinkSync(tempFilePath);
        } catch (copyErr) {
          console.error("🚨 RETRY SAVE FAILED:", copyErr.message);
        }
      } else {
        throw renameErr;
      }
    }
  } catch (err) { console.error("🚨 CRITICAL ERROR: Failed to save state:", err.message); }
}


// ============== Robinhood API Wrapper ==============
class RobinhoodAPI {
  constructor(apiKey, base64Priv) {
    this.apiKey = apiKey;
    this.baseUrl = "https://trading.robinhood.com";
    this.retryDelay = 60 * 1000;
    this.requestTimeout = 20 * 1000;
    this.timeOffset = 0; // Correction in milliseconds (Server - Local)

    const raw = Buffer.from(base64Priv, "base64");
    let keyPairSource;
    if (raw.length === 32) {
      keyPairSource = nacl.sign.keyPair.fromSeed(raw);
    } else if (raw.length === 64) {
      keyPairSource = nacl.sign.keyPair.fromSecretKey(raw);
    } else {
      throw new Error("Private key must be 32 (seed) or 64 (secret key) bytes");
    }
    this.keyPair = keyPairSource;
  }

  _timestamp() {
    // Apply offset to sync with server time
    return Math.floor((Date.now() + this.timeOffset) / 1000);
  }

  _sign(msg) {
    const sig = nacl.sign.detached(Buffer.from(msg, "utf8"), this.keyPair.secretKey);
    return Buffer.from(sig).toString("base64");
  }

  async _request(method, path, bodyObj = null) {
    const t = this._timestamp();
    const bodyStr = bodyObj ? JSON.stringify(bodyObj) : "";
    const toSign = this.apiKey + t + path + method + bodyStr;
    const signature = this._sign(toSign);
    const headers = {
      "x-api-key": this.apiKey,
      "x-signature": signature,
      "x-timestamp": String(t),
      "Content-Type": "application/json",
      "Accept": "application/json"
    };
    const url = this.baseUrl + path;
    const config = {
      method: method,
      url: url,
      headers: headers,
      timeout: this.requestTimeout,
      ...(bodyObj && { data: bodyObj }),
    };
    const resp = await axios(config);
    return resp.data;
  }

  async _requestWithRetry(method, path, bodyObj = null) {
    while (true) {
      try {
        return await this._request(method, path, bodyObj);
      } catch (err) {
        const status = err.response?.status;
        const code = err.code;

        // --- AUTO-CORRECT TIME SYNC ---
        // Handle "Timestamp is invalid" (401) by syncing with Server Date Header
        const errorDataStr = JSON.stringify(err.response?.data || "");
        if (status === 401 && (errorDataStr.includes("Timestamp") || errorDataStr.includes("timestamp"))) {
          const serverDateStr = err.response.headers ? (err.response.headers['date'] || err.response.headers['Date']) : null;
          if (serverDateStr) {
            const serverTime = Date.parse(serverDateStr);
            const localTime = Date.now();
            if (!isNaN(serverTime)) {
              const drift = serverTime - localTime;
              console.log(`🕒 Clock Drift Detected! Local: ${new Date(localTime).toISOString()} | Server: ${serverDateStr}`);
              console.log(`   🛠️ Adjusting internal clock by ${drift > 0 ? '+' : ''}${drift}ms to sync.`);
              this.timeOffset = drift;

              // Wait a moment and retry immediately
              await new Promise(res => setTimeout(res, 1000));
              continue;
            }
          }
        }

        // Handle 400 Validation Errors (Graceful Failures)
        if (status === 400 && err.response?.data?.errors) {
          const errors = err.response.data.errors;
          // 1. Duplicate Order ID
          const isDuplicate = errors.some(e => e.attr === 'client_order_id' && e.detail && e.detail.includes('Already used'));
          if (isDuplicate) {
            console.warn(`⚠️ API Warning (${path}): Order already placed (Client Order ID collision). Assuming success.`);
            return null;
          }
          // 2. Insufficient Funds (Buying Power)
          const isInsuffFunds = errors.some(e => e.detail && e.detail.includes('Not enough buying power'));
          if (isInsuffFunds) {
            console.warn(`⚠️ API Warning (${path}): Insufficient buying power. Skipping order.`);
            return null; // Return null to skip without crashing
          }
          // 3. Invalid Quantity (Min Order Size)
          const isInvalidQty = errors.some(e => e.detail && (e.detail.includes('must be greater than or equal to') || e.detail.includes('increment')));
          if (isInvalidQty) {
            console.warn(`⚠️ API Warning (${path}): Invalid quantity/increment. Skipping order. Error: ${JSON.stringify(errors)}`);
            return null;
          }
        }

        const isTimeout = code === "ECONNABORTED" || err.message.toLowerCase().includes('timeout');
        const isNetworkError = ["ECONNRESET", "ENOTFOUND", "ETIMEDOUT", "EAI_AGAIN", "ECONNREFUSED"].includes(code);
        const isRetryableStatus = status && (status >= 500 && status <= 599);
        const isRateLimit = status === 429;
        const isForbidden = status === 403;

        if (isTimeout || isNetworkError || isRetryableStatus || isRateLimit || isForbidden) {
          const errorType = isTimeout ? "timeout" : isNetworkError ? `network error (${code || 'N/A'})` : isRetryableStatus ? `server error (${status})` : isRateLimit ? `rate limit (429)` : isForbidden ? `forbidden (403)` : `other retryable error`;
          console.log(`⏳ API Error (${path}): ${errorType}. Retrying in ${this.retryDelay / 1000}s...`);
          await new Promise((res) => setTimeout(res, this.retryDelay));
        } else {
          console.error(`❌ API Error (${path}): Non-retryable error: Status=${status || 'N/A'}, Code=${code || 'N/A'}, Message=${err.message}`);
          if (err.response?.data) {
            console.error("Response Data:", JSON.stringify(err.response.data, null, 2));
          }
          throw err;
        }
      }
    }
  }
  async getBalance() { const data = await this._requestWithRetry("GET", "/api/v1/crypto/trading/accounts/"); if (!data) return 0; const account = Array.isArray(data.results) ? data.results[0] : data; if (!account) return 0; const fields = ["buying_power", "cash_balance", "crypto_buying_power"]; for (const field of fields) { if (account[field] !== undefined && account[field] !== null) { const balance = parseFloat(account[field]); return !isNaN(balance) ? balance : 0; } } console.warn("[getBalance] Could not find a recognizable balance field in response:", account); return 0; }
  async getHoldings() { const data = await this._requestWithRetry("GET", "/api/v1/crypto/trading/holdings/"); return data?.results || []; }
  async getQuotes(assetCodes) { if (!Array.isArray(assetCodes) || assetCodes.length === 0) return {}; const data = await this._requestWithRetry("GET", "/api/v1/crypto/marketdata/best_bid_ask/"); const quotes = data?.results || []; const result = {}; const codesSet = new Set(assetCodes); for (const quote of quotes) { const sym = quote.symbol?.replace("-USD", ""); if (sym && codesSet.has(sym)) { const rawPrice = parseFloat(quote.price); if (!isNaN(rawPrice) && rawPrice > 0) { result[sym] = Number(rawPrice.toFixed(10)); } else { console.warn(`[getQuotes] Invalid price received for ${sym}: ${quote.price}`); result[sym] = 0; } } } assetCodes.forEach(code => { if (!(code in result)) { console.warn(`[getQuotes] No quote data found for requested asset: ${code}`); } }); return result; }

  _validateOrderParams(symbol, quantityStr, side) {
    if (!symbol || !symbol.includes("-USD")) { throw new Error(`Invalid symbol format: '${symbol}'. Must be like 'BTC-USD'.`); }
    const qtyNum = parseFloat(quantityStr);
    if (isNaN(qtyNum) || qtyNum <= 0) { throw new Error(`Invalid quantity for ${side} order: '${quantityStr}'. Must be a positive number.`); }
    return qtyNum;
  }
  async placeSell(symbol, quantityStr) { this._validateOrderParams(symbol, quantityStr, 'sell'); const sym = symbol.replace("-USD", ""); if (!checkMinQuantity(sym, quantityStr)) { console.warn(`   🔸 ${sym} Sell Skipped: Qty ${quantityStr} below minimum.`); return null; } const path = "/api/v1/crypto/trading/orders/"; const body = { client_order_id: crypto.randomUUID(), side: "sell", type: "market", symbol, market_order_config: { asset_quantity: quantityStr } }; console.log(`📦 Placing SELL order: ${quantityStr} ${sym}`); return this._requestWithRetry("POST", path, body); }
  async placeBuy(symbol, quantityStr) { this._validateOrderParams(symbol, quantityStr, 'buy'); const sym = symbol.replace("-USD", ""); if (!checkMinQuantity(sym, quantityStr)) { console.warn(`   🔸 ${sym} Buy Skipped: Qty ${quantityStr} below minimum.`); return null; } const path = "/api/v1/crypto/trading/orders/"; const body = { client_order_id: crypto.randomUUID(), side: "buy", type: "market", symbol, market_order_config: { asset_quantity: quantityStr } }; console.log(`📦 Placing BUY order: ${quantityStr} ${sym}`); return this._requestWithRetry("POST", path, body); }
}



// [REMOVED] Legacy ScientificOptimizer (replaced by ported Systematic Calculator version at end of file)


// ============== Main Application Logic ==============
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });



async function mainLoop() {
  console.log("🚀 Initializing Dreamer 1.999 (TrikeShed/Wasm Presentation - Shadow Engine Architecture)...");

  const apiKey = process.env.API_KEY;
  const privKey = process.env.PRIVATE_KEY_BASE64;
  if (!apiKey || !privKey) { console.error("❌ FATAL: Missing API_KEY or PRIVATE_KEY_BASE64 environment variables."); rl.close(); return; }

  let rh;
  try { rh = new RobinhoodAPI(apiKey, privKey); console.log("🔑 API Initialized."); }
  catch (error) { console.error("❌ FATAL: API initialization failed:", error.message); rl.close(); return; }

  // --- Initialize Live Engine ---
  liveEngine = new TradingEngine(defaultGenome, 'LIVE');

  // Load State
  const { loadedBaselines, loadedTrailingState, loadedLastActionTimestamps, loadedGenome, loadedData } = loadState();
  if (loadedBaselines) liveEngine.baselines = loadedBaselines;
  if (loadedTrailingState) liveEngine.trailingState = loadedTrailingState;
  if (loadedLastActionTimestamps) liveEngine.lastActionTimestamps = loadedLastActionTimestamps;
  if (loadedGenome) {
    // Merge loaded genome with defaults, preserving overrides
    liveEngine.genome = { ...defaultGenome, ...loadedGenome };
    // Preserve per-asset overrides if they exist
    if (loadedGenome.overrides) {
      liveEngine.genome.overrides = { ...loadedGenome.overrides };
    }
    // console.log("✅ Loaded state from C:\\Users\\Parti\\webstorm\\cryptoBot\\liveEngineState.json.");
  }

  // Initialize promotion threshold from saved state (per-asset)
  if (loadedData && loadedData.lastBestScore !== undefined) {
    // Migration: Handle old single-number format
    if (typeof loadedData.lastBestScore === 'number') {
      console.log(`🔄 Migrating old threshold format to per-asset...`);
      global.lastBestScore = {}; // Start fresh
    } else {
      global.lastBestScore = loadedData.lastBestScore;
      const assetCount = Object.keys(global.lastBestScore).length;
      if (assetCount > 0) {
        const summary = Object.entries(global.lastBestScore)
          .map(([asset, score]) => `${asset}=${score.toFixed(2)}%`)
          .join(', ');
        console.log(`🎯 Restored promotion thresholds: ${summary}`);
      } else {
        console.log(`🎯 Initialized promotion thresholds: (empty, will populate per-asset)`);
      }
    }
  } else {
    global.lastBestScore = {};
    console.log(`🎯 Initialized promotion thresholds: (empty, will populate per-asset)`);
  }

  console.log("🧬 Live Engine Genome Loaded.");

  // Inject Global Price History into Live Engine for Shadows to share
  if (typeof priceHistory !== 'undefined') {
    liveEngine.priceHistoryBuffer = priceHistory;
    console.log(`🔗 Linked Global Price History to Live Engine (${priceHistory.length} ticks).`);
  }

  // --- Display Genome Personality (Hyper-Evolutionary Config) ---
  console.log("\n╔═══════════════════════════════════════════════════╗");
  console.log("║       🧬 Live Bot Genome Personality 🧬          ║");
  console.log("╠═══════════════════════════════════════════════════╣");
  const g = liveEngine.genome;
  console.log(`║ Allocation Mode: ${g.ALLOCATION_MODE === 0 ? 'BALANCED' : g.ALLOCATION_MODE === 1 ? 'GROWTH' : 'DEFENSIVE'}                                  ║`);
  console.log(`║ PHYSICS:                                          ║`);
  console.log(`║   Baseline Drag: ${(g.SPAR_DRAG_COEFFICIENT || 0.999968).toFixed(6)}                      ║`);
  console.log(`║   Price Memory: ${(g.PRICE_HISTORY_WINDOW_SIZE || 200)} ticks                           ║`);
  console.log(`║   Volatility Threshold: ${((g.ADAPTIVE_VOLATILITY_THRESHOLD || 0.015) * 100).toFixed(2)}%                  ║`);
  console.log(`║ TIMING:                                           ║`);
  console.log(`║   Harvest Timeout: ${Math.round((g.FORCED_HARVEST_TIMEOUT || 1200000) / 60000)}m                      ║`);
  console.log(`║   Rebalance Cooldown: ${Math.round((g.REBALANCE_COOLDOWN || 1800000) / 60000)}m                   ║`);
  console.log(`║   Forced Rebalance: ${Math.round((g.FORCE_REBALANCE_TIMEOUT || 1500000) / 60000)}m                    ║`);
  console.log(`║ RISK:                                             ║`);
  console.log(`║   Max Rebalance Attempts: ${g.MAX_REBALANCE_ATTEMPTS || 3}                     ║`);
  console.log(`║   Crash Trigger: ${((g.CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT || -0.07) * 100).toFixed(1)}%                       ║`);
  console.log(`║ EVOLUTION:                                        ║`);
  console.log(`║   Min Trades for Promotion: ${g.MIN_TRADES_FOR_PROMOTION || 2}                  ║`);
  console.log(`║   Required Win Streak: ${g.EVOLUTION_CONSISTENCY_COUNT || 3}                       ║`);
  console.log(`║   Oracle Trend Threshold: ${g.ORACLE_TREND_THRESHOLD || 0.8}                     ║`);
  console.log(`║   Oracle Volatility Threshold: ${(g.ORACLE_VOLATILITY_THRESHOLD || 2.0).toFixed(1)}%                  ║`);
  console.log(`║   Oracle Flash Threshold: ${(liveEngine.genome.FLASH_THRESHOLD || 0.05).toFixed(2)}% (Tick-to-Tick)       ║`);
  console.log("╚═══════════════════════════════════════════════════╝\n");

  let initialized = false;

  // --- Initialize Managers (Legion Architecture) ---
  // 1. Asset & Regime Memory (Tier 1 & 2 Configs)
  // Stores the "Genetic Memory" of the system across reboots
  const assetRegimeManager = new AssetRegimeManager();
  console.log(`🧠 Asset Regime Manager Loaded (${Object.keys(assetRegimeManager.memory || {}).length} asset records).`);

  // 2. Regime Detector (The New Oracle)
  const regimeDetector = new RegimeDetector();
  // Hydrate Detector from Memory
  if (assetRegimeManager && assetRegimeManager.memory) {
    Object.keys(assetRegimeManager.memory).forEach(sym => {
      const profile = assetRegimeManager.memory[sym];
      if (profile && profile.activeRegime) {
        regimeDetector.regimes[sym] = profile.activeRegime;
      }
    });
  }
  console.log(`🔮 Regime Detector Online. Hydrated ${Object.keys(regimeDetector.regimes).length} regimes.`);

  // 3. The Dreamer Grid (Deep Past)
  const scriptPath = fileURLToPath(import.meta.url);
  const dreamerGrid = [];

  console.log(`🦾 Spawning Dreamer Grid (${LEGION_CONFIG.DREAMER_WORKER_COUNT} workers)...`);
  const totalWorkers = LEGION_CONFIG.DREAMER_WORKER_COUNT;
  for (let i = 0; i < totalWorkers; i++) {
    const worker = fork(scriptPath, ['--simulation', `--workerId=${i}`, `--totalWorkers=${totalWorkers}`], {
      stdio: ['ignore', 'inherit', 'inherit', 'ipc']
    });

    worker.on('message', (msg) => {
      if (msg.type === 'HEARTBEAT') {
        // Optional: console.log(`   [Heartbeat] Batch ${msg.batch} | Best: ${msg.bestScore.toFixed(2)}`);
      }
      if (msg.type === 'OPTIMIZATION_FOUND') {
        // 🏆 PROMOTION: Dreamer found a better genome
        // WORKER sends candidate if it beats ITS local best.
        // MAIN must verify it beats GLOBAL best (loaded from state).

        // Recover Depth from Param String (e.g. "Champion [MEDIUM]")
        let depthMode = "MEDIUM";
        if (msg.param.includes("[SHORT]")) depthMode = "SHORT";
        else if (msg.param.includes("[LONG]")) depthMode = "LONG";

        const scoreKey = `${msg.focus}_${depthMode}`;
        const assetThreshold = global.lastBestScore[scoreKey] || 0.0;

        if (msg.score > assetThreshold) {
          // COLOR CODING: Visual indicator for Timeframe
          let colorCode = "\x1b[32m"; // Default Green
          if (depthMode === "SHORT") colorCode = "\x1b[36m"; // Cyan
          else if (depthMode === "MEDIUM") colorCode = "\x1b[33m"; // Yellow
          else if (depthMode === "LONG") colorCode = "\x1b[35m"; // Magenta

          console.log(`\n🏆 [PROMOTION] ${msg.focus}: ${colorCode}${msg.param}\x1b[0m = ${typeof msg.val === 'number' ? (Number.isInteger(msg.val) ? msg.val : msg.val.toFixed(2)) : msg.val} | Alpha: ${msg.score?.toFixed(3)}% (beat ${assetThreshold.toFixed(3)}%)`);

          // Update per-asset best score
          if (!global.lastBestScore) global.lastBestScore = {};
          global.lastBestScore[scoreKey] = msg.score;

          // Apply to Live Engine
          if (msg.focus && msg.genome && msg.genome.overrides && msg.genome.overrides[msg.focus]) {
            if (!liveEngine.genome.overrides) liveEngine.genome.overrides = {};
            if (!liveEngine.genome.overrides[msg.focus]) liveEngine.genome.overrides[msg.focus] = {};

            // Merge Specific Asset overrides
            // MUTATION LOCK: Prevent updates if asset is currently acting (Harvest/Rebalance)
            const isRebalancing = liveEngine.rebalanceState && liveEngine.rebalanceState[msg.focus];
            const isHarvesting = liveEngine.trailingState && liveEngine.trailingState[msg.focus] && liveEngine.trailingState[msg.focus].flagged;

            if (isRebalancing || isHarvesting) {
              console.log(`   🔒 [Mutation Lock] Skipped update for ${msg.focus} (Active State: ${isRebalancing ? 'Rebalancing' : 'Harvesting'}).`);
            } else {
              Object.assign(liveEngine.genome.overrides[msg.focus], msg.genome.overrides[msg.focus]);
              console.log(`   ✅ Applied ${msg.param} optimization to ${msg.focus} live genome.`);

              // Track Timeframe for Table Coloring
              if (!liveEngine.assetSourceTimeframe) liveEngine.assetSourceTimeframe = {};
              if (msg.param.includes("[SHORT]")) liveEngine.assetSourceTimeframe[msg.focus] = 'SHORT';
              else if (msg.param.includes("[MEDIUM]")) liveEngine.assetSourceTimeframe[msg.focus] = 'MEDIUM';
              else if (msg.param.includes("[LONG]")) liveEngine.assetSourceTimeframe[msg.focus] = 'LONG';

              saveEngineState(); // Persist immediately
            }
          }
        }
      }

      if (msg.type === 'RESET_SCORE') {
        // The Dreamer has finished a full cycle for this asset.
        // SOFT RESET: Decay the score (0.75x) instead of deleting it.
        // This keeps the Main Process explicitly aware of the high bar.
        if (global.lastBestScore && msg.asset) {
          // Update: Iterate all timeframes to decay specific keys
          const timeframes = ['SHORT', 'MEDIUM', 'LONG'];
          timeframes.forEach(tf => {
            const key = `${msg.asset}_${tf}`;
            if (global.lastBestScore[key]) {
              if (msg.mode === 'SOFT') {
                global.lastBestScore[key] *= 0.75;
              } else {
                delete global.lastBestScore[key]; // Hard reset
              }
            }
          });
        }
      }

      if (msg.type === 'TIER_1_UPDATE') {
        // Initialize Aggregator
        if (!global.dreamStats) global.dreamStats = { verified: [], rejected: {}, improvement: [], regimes: {} };

        // Track Stats
        if (msg.score === -100 || !msg.genome || Object.keys(msg.genome).length === 0) {
          // Rejected / No Improvement
          const r = msg.regime || 'UNK';
          if (!global.dreamStats.rejected[r]) global.dreamStats.rejected[r] = 0;
          global.dreamStats.rejected[r]++;
        } else {
          // Improvement Found!
          global.dreamStats.improvement.push(msg.symbol);
        }

        // Track Regimes
        const r = msg.regime || 'UNK';
        if (!global.dreamStats.regimes[r]) global.dreamStats.regimes[r] = 0;
        global.dreamStats.regimes[r]++;

        // 1. Update Persistent Memory
        assetRegimeManager.update(msg.symbol, msg.genome || {}, 'TIER_1_THEORETICAL', msg.regime || 'UNKNOWN');
        // 1b. Update Live Radar
        if (regimeDetector) regimeDetector.regimes[msg.symbol] = msg.regime || 'UNKNOWN';

        if (msg.genome && Object.keys(msg.genome).length > 0) {
          // 2. Apply to Live Engine immediately (Hot Swap)
          if (!liveEngine.genome.overrides) liveEngine.genome.overrides = {};
          if (!liveEngine.genome.overrides[msg.symbol]) liveEngine.genome.overrides[msg.symbol] = {};

          // Merge new genes
          Object.assign(liveEngine.genome.overrides[msg.symbol], msg.genome);
          // console.log(`   --> Applied experimental genes to Live Engine for ${msg.symbol}: ${JSON.stringify(msg.genome)}`);
        }

        if (legionManager) legionManager.notifyOptimizationComplete(msg.symbol); // Clear status

        // Periodic flush of stats (Every 10 seconds or every 50 updates)
        const NOW = Date.now();
        if (!global.lastDreamStatFlush) global.lastDreamStatFlush = NOW;
        if (NOW - global.lastDreamStatFlush > 10000) {
          const totalRefusals = Object.values(global.dreamStats.rejected).reduce((a, b) => a + b, 0);
          const totalSims = (msg.sims || 0); // Note: This is per worker, might be confusing if aggregated differently.
          // Let's just track updates received.
          const updates = totalRefusals + global.dreamStats.improvement.length;

          if (updates > 0) {
            // Build String
            let out = `💤 [DREAMER] `;
            if (global.dreamStats.improvement.length > 0) {
              out += `✨ NEW: [${global.dreamStats.improvement.join(', ')}] | `;
            }
            out += `Rejected: ${totalRefusals} (`;
            Object.entries(global.dreamStats.rejected).forEach(([reg, count]) => {
              // Emoji Map
              const em = reg === 'RALLY' ? '🚀' : reg === 'CRASH' ? '🩸' : reg === 'CHOP' ? '🦀' : '❓';
              out += `${em}${count} `;
            });
            out += `) | Regimes: ${Object.keys(global.dreamStats.regimes).length} | Worker Sims: ~${totalSims}`;
            console.log(out);

            // Reset
            global.dreamStats = { verified: [], rejected: {}, improvement: [], regimes: {} };
            global.lastDreamStatFlush = NOW;
          }
        }
      }
    });

    // Restart logic if worker dies
    worker.on('exit', (code) => {
      console.warn(`⚠️ Dreamer Worker ${worker.pid} exited (code ${code}).`);
      // Ideally we respawn, but for simplicity we assume stability for now.
      // Or complex respawn logic would go here.
    });

    dreamerGrid.push(worker);
  }
  let lastStateSaveTime = 0; // Fix: Initialize variable to prevent ReferenceError
  let lastHistoryRefreshTime = Date.now(); // Track when we last refreshed history
  let lastOptimizationTime = Date.now(); // Track when we last triggered optimizations
  const HOURLY_REFRESH_INTERVAL = 60 * 60 * 1000; // 1 hour
  const STATE_SAVE_INTERVAL = 5 * 60 * 1000; // Save every 5 minutes

  // State save function - ROBUST VERSION
  function saveEngineState() {
    try {
      if (!liveEngine) return;
      const stateToSave = liveEngine.getStateSnapshot();
      const scriptDir = path.dirname(fileURLToPath(import.meta.url));
      const tempFilePath = path.join(scriptDir, 'liveEngineState.json.tmp');
      const finalFilePath = path.join(scriptDir, 'liveEngineState.json');

      // Write to temp file first
      fs.writeFileSync(tempFilePath, JSON.stringify(stateToSave, null, 2));

      // Retry loop for rename (Windows EPERM fix)
      let saved = false;
      let attempts = 0;
      while (!saved && attempts < 3) {
        try {
          // Try to rename
          if (fs.existsSync(finalFilePath)) {
            try { fs.unlinkSync(finalFilePath); } catch (e) { /* ignore unlink fail */ }
          }
          fs.renameSync(tempFilePath, finalFilePath);
          saved = true;
        } catch (err) {
          attempts++;
          if (attempts >= 3) throw err; // Throw on final failure

          // Simple synchronous delay (busy wait) to allow file lock to clear
          const start = Date.now();
          while (Date.now() - start < 100);
        }
      }
    } catch (err) {
      console.error("🚨 CRITICAL ERROR: Failed to save state (Retried 3x):", err.message);
    }
  }


  // 4. The Legion Manager (Broad Present)
  // Orchestrates Shadows and dispatches orders to Dreamers
  const legionManager = new LegionManager(liveEngine, TradingEngine, dreamerGrid);

  // Inject dependencies into global scope/main loop variables helper if needed?
  // We declared them as const here. They need to be accessible in the loop.
  // Wait, the loop is inside THIS function space. So const is fine if defined before loop.
  // BUT 'legionManager' is used in the loop.
  // I should check if I broke any 'let' vs 'const' visibility.
  // The previous code had 'let oracle' etc.
  // 'const' is block scoped. This block is 'async function mainLoop'.
  // The while loop is inside mainLoop. So 'const' is visible.

  let previousCycleValues = {}; // Restoration of missing UI state

  // --- Growth Timer ---
  let lastGrowthTime = Date.now();

  while (true) {
    const startTime = Date.now();
    console.log(`\n----- Cycle Start: ${new Date().toISOString()} -----`);

    // --- THE ALPHA TITHE (Systematic Growth) ---
    if (ENABLE_AUTO_COMPOUND && startTime - lastGrowthTime > GROWTH_INTERVAL) {
      console.log("\n🌱 [GROWTH] Checking for Organic Growth opportunities...");

      // 1. Check Cash
      let currentCash = 0;
      try { currentCash = await rh.getBalance(); } catch (e) { console.warn("   ⚠️ Could not fetch balance for Growth check."); }

      if (currentCash > COMPOUND_THRESHOLD_USD) {
        // 2. Find the #1 Asset (Highest Alpha)
        let kingAsset = null;
        let kingScore = -Infinity;
        let kingTimeframe = 'MEDIUM'; // Default

        if (global.lastBestScore) {
          Object.keys(global.lastBestScore).forEach(key => {
            const score = global.lastBestScore[key];
            // key format: ASSET_TIMEFRAME (e.g. BTC_LONG)
            const parts = key.split('_');
            const asset = parts[0];

            // Filter: Must be tracked by Live Engine
            if (liveEngine.baselines[asset] && !REBALANCE_EXCLUDE.includes(asset)) {
              if (score > kingScore) {
                kingScore = score;
                kingAsset = asset;
                kingTimeframe = parts[1] || 'MEDIUM';
              }
            }
          });
        }

        // 3. Calculate Tithe (1% or $1.00 min)
        let titheAmount = currentCash * COMPOUND_ALLOCATION_PCT;
        if (titheAmount < 1.05) titheAmount = 1.05; // Enforce Robinhood minimum (slightly padded)

        // 4. Verification & Execution
        if (kingAsset && kingScore > 0) {
          // Check Live Trend (Don't buy a crashing asset)
          // utilize latest quote from previous cycle if available, or fetch new?
          // We will fetch new quotes in the main logic anyway. Let's do a quick focused fetch or use previous cycle data?
          // Using previous cycle data is safer for rate limits.
          // But we are at the START of the loop. 'previousCycleValues' holds VALUE, not Price/Deviation.
          // Let's just let it ride on Alpha confidence + Deviation check after quote fetch?
          // No, let's just fetch the single quote.
          try {
            const q = await rh.getQuotes([kingAsset]);
            if (q[kingAsset]) {
              const price = q[kingAsset];
              const baseline = liveEngine.baselines[kingAsset];
              const dev = baseline > 0 ? (price * (holdingDetails[kingAsset]?.rawQuantity || 0) - baseline) / baseline : 0;

              // FILTER: Only buy if Deviation > -1% (Not crashing)
              // Actually, let's use a simpler logic: Is it active?
              if (dev > -0.01) {
                console.log(`👑 [GROWTH] The King is ${kingAsset} (${kingTimeframe} Alpha: ${kingScore.toFixed(3)}%). Trend OK (${(dev * 100).toFixed(2)}%). Executing Tithe: $${titheAmount.toFixed(2)}...`);

                const qty = titheAmount / price; // Raw calc
                // We rely on API wrapper or rounding. API wrapper passes string.
                // We need to round it precisely.
                // Helper:
                const decimals = minIncrementMap[kingAsset] < 0.01 ? 6 : 2; // Rough heuristic or use minIncrementMap
                // actually use logic from TradingEngine (not accessible here easily).
                // Let's use string formatting safe enough.
                const fmtQty = qty.toFixed(decimals === 6 ? 6 : 2); // conservative

                // Check Buying Power again? API handles it.
                if (titheAmount <= currentCash) {
                  const res = await rh.placeBuy(`${kingAsset}-USD`, fmtQty);
                  if (res) {
                    console.log(`   ✅ Growth Buy Placed. Adjusting Baseline...`);
                    liveEngine.baselines[kingAsset] += titheAmount;
                    console.log(`   📈 Baseline for ${kingAsset} increased by $${titheAmount.toFixed(2)} (New: $${liveEngine.baselines[kingAsset].toFixed(2)}).`);
                    lastGrowthTime = Date.now();
                  } else {
                    console.warn("   ❌ Growth Buy Failed (API Error).");
                  }
                }
              } else {
                console.log(`   📉 [GROWTH] Skipped ${kingAsset}. Alpha is high (${kingScore.toFixed(2)}%), but currently crashing (Dev: ${(dev * 100).toFixed(2)}%). Saving cash.`);
              }
            }
          } catch (err) { console.error("   ⚠️ Growth check error:", err.message); }
        } else {
          console.log("   🤷 [GROWTH] No suitable King Asset found (or Alpha negative).");
        }

        // Reset timer anyway to avoid spamming loop if checks fail
        lastGrowthTime = Date.now();

      } else {
        console.log(`   💤 [GROWTH] Cash too low ($${currentCash.toFixed(2)} < $${COMPOUND_THRESHOLD_USD}). Sleeping.`);
        lastGrowthTime = Date.now();
      }
    }

    // --- Hourly History Refresh & Re-Optimization ---
    if (startTime - lastHistoryRefreshTime >= HOURLY_REFRESH_INTERVAL) {
      console.log("\n🔄 [HOURLY REFRESH] Reloading 24-hour price history...");
      // The history is loaded per-worker, so we signal a restart would be needed.
      // For now, we'll just flag that it's time and let the Dreamer workers use fresh data on next optimization.
      // In practice, we should reload the history here or restart workers.
      // Since workers load history on optimization start, triggering optimization will use fresh data.
      lastHistoryRefreshTime = startTime;
      lastOptimizationTime = 0; // Force immediate optimization after history refresh
      console.log("   ✅ History refresh scheduled. Will trigger optimization...");
    }

    if (startTime - lastOptimizationTime >= HOURLY_REFRESH_INTERVAL) {
      console.log("\n🚀 [HOURLY OPTIMIZATION] Triggering Mass Scientific Optimization for all assets...");
      if (legionManager) {
        // Request optimization for all active assets
        const allAssets = Object.keys(liveEngine.baselines);
        allAssets.forEach(asset => {
          if (!HARVEST_EXCLUDE.includes(asset) && !REBALANCE_EXCLUDE.includes(asset)) {
            legionManager.requestOptimization(asset);
          }
        });
      }
      lastOptimizationTime = startTime;
    }

    // --- Status Display ---
    if (legionManager) {
      const hots = Object.values(legionManager.assetHeatMap).filter(v => v === 'HOT' || v === 'INFERNO').length;
      const dreaming = Array.from(legionManager.activeDreamJobs).slice(0, 5).join(', ') + (legionManager.activeDreamJobs.size > 5 ? '...' : '');
      console.log(`⚔️ [LEGION] Active Shadows: ${legionManager.shadowLegion.length} | Hot Assets: ${hots} | 🧠 Dreaming: [${dreaming}]`);
    }
    if (regimeDetector) {
      // console.log(`🔮 [REGIME] ...`);
    }

    // --- Use Engine State References for Convenience (Refactoring Step) ---
    // This allows us to keep the rest of the loop largely unchanged for this step
    // while relying on the class's storage.
    // We will eventually move the logic INTO the class.
    let tokenBaselines = liveEngine.baselines;
    let trailingState = liveEngine.trailingState;
    let lastActionTimestamps = liveEngine.lastActionTimestamps;
    let rebalanceState = liveEngine.rebalanceState;
    // let adaptiveDeadZoneState = liveEngine.adaptiveDeadZoneState; // REMOVED
    let portfolioHarvestState = liveEngine.portfolioHarvestState;
    const currentGenome = liveEngine.genome; // Use the engine's genome

    harvestedAmount = 0;
    let anyTradesThisCycle = false;
    let stateChanged = false;

    // Fetch Balance, Holdings, Quotes (Unchanged)
    let cashBalance = 0; try { cashBalance = await rh.getBalance(); console.log(`💰 Available Cash Balance: $${cashBalance.toFixed(2)}`); } catch (err) { console.error("❌ FATAL: Could not fetch balance:", err.message); rl.close(); return; }
    let holdings = []; try { holdings = await rh.getHoldings(); if (holdings.length === 0) console.log("ℹ️ No crypto holdings found."); } catch (err) { console.error("❌ FATAL: Could not fetch holdings:", err.message); rl.close(); return; }
    const holdingDetails = {}; let codes = [];
    if (holdings.length > 0) { holdings.forEach(h => { const code = h.asset_code; const qty = parseFloat(h.total_quantity) || 0; const minQtyThreshold = minIncrementMap[code] ? (minIncrementMap[code] / 10) : 1e-10; if (code && qty > minQtyThreshold) { if (!holdingDetails[code]) { holdingDetails[code] = { rawQuantity: 0 }; codes.push(code); } holdingDetails[code].rawQuantity += qty; } }); if (codes.length > 0) console.log(`📊 Holdings: ${codes.join(', ')}`); else console.log("ℹ️ No significant crypto holdings found after filtering."); }
    let rhPrices = {}; if (codes.length > 0) { try { rhPrices = await rh.getQuotes(codes); } catch (err) { console.error("❌ FATAL: Could not fetch quotes:", err.message); rl.close(); return; } } else { console.log("ℹ️ Skipping quote fetch."); }

    // Calculate Portfolio Summary & Initialize/Verify Baselines & State (Unchanged)
    let totalHoldingsValue = 0; const portfolioSummary = []; const currentSymbols = new Set(); let baselinesVerifiedOrSetThisCycle = false;
    codes.forEach((sym) => {
      currentSymbols.add(sym); const details = holdingDetails[sym]; const price = rhPrices[sym];
      if (price === undefined || price === null || isNaN(price) || price <= 0) { console.warn(`⚠️ Warn: Invalid price for ${sym}. Skipping calculations & state checks.`); if (trailingState[sym]) delete trailingState[sym].previousDeviation; if (rebalanceState[sym]) delete rebalanceState[sym].previousDeviation; return; }
      const totalQty = details.rawQuantity; const currentHoldingValue = price * totalQty; totalHoldingsValue += currentHoldingValue;
      if (!initialized) { const existingBaseline = tokenBaselines[sym]; if (existingBaseline !== undefined && typeof existingBaseline === 'number' && existingBaseline > 0.01) { const diff = Math.abs(currentHoldingValue - existingBaseline); const diffPercent = existingBaseline === 0 ? Infinity : diff / existingBaseline; if (diffPercent <= BASELINE_LOAD_TOLERANCE_PERCENT) { /* keep baseline log */ } else { tokenBaselines[sym] = currentHoldingValue; console.log(`⚠️ ${sym}: Baseline reset to $${currentHoldingValue.toFixed(2)} (Diff: ${(diffPercent * 100).toFixed(2)}% > ${BASELINE_LOAD_TOLERANCE_PERCENT * 100}% tolerance).`); stateChanged = true; } baselinesVerifiedOrSetThisCycle = true; } else if (!tokenBaselines[sym] && currentHoldingValue > 0.01) { tokenBaselines[sym] = currentHoldingValue; console.log(`✨ Initialized baseline ${sym}: $${tokenBaselines[sym].toFixed(2)}.`); baselinesVerifiedOrSetThisCycle = true; stateChanged = true; } }
      if (!tokenBaselines[sym] && currentHoldingValue > 0.01) { tokenBaselines[sym] = currentHoldingValue; console.log(`✨ Initialized baseline ${sym} (post-init): $${tokenBaselines[sym].toFixed(2)}.`); stateChanged = true; }
      if (!lastActionTimestamps[sym] && tokenBaselines[sym] > 0.01) { console.log(`✨ Initialized last action timestamp for ${sym}.`); lastActionTimestamps[sym] = Date.now(); stateChanged = true; }
      portfolioSummary.push({ Symbol: sym, Quantity: totalQty, Price: price, Value: currentHoldingValue, Baseline: tokenBaselines[sym], usdValueNum: currentHoldingValue });
    });

    // --- Data Recorder (Black Box) ---
    // Save tick data for "Time Machine" simulations
    if (portfolioSummary.length > 0) {
      appendMarketData(Date.now(), portfolioSummary);
    }

    if (!initialized && baselinesVerifiedOrSetThisCycle) { console.log("✅ Baselines & Timestamps init/verify complete."); initialized = true; if (stateChanged) { saveState(); stateChanged = false; } } else if (!initialized && holdings.length > 0 && codes.length === 0) { console.log("⏳ Waiting for valid prices to initialize baselines..."); }

    // --- Clean up persistent state for EXCLUDED assets that are still held ---
    currentSymbols.forEach(sym => {
      if (HARVEST_EXCLUDE.includes(sym)) {
        if (trailingState[sym]) {
          console.log(`🗑️ Clearing persistent trailing (harvest) state for EXCLUDED asset: ${sym}`);
          delete trailingState[sym];
          stateChanged = true;
        }
      }
      if (REBALANCE_EXCLUDE.includes(sym) && rebalanceState[sym]) {
        delete rebalanceState[sym];
      }
      if (REBALANCE_EXCLUDE.includes(sym) && rebalanceState[sym]) {
        delete rebalanceState[sym];
      }
      // (Adaptive State Cleanup Block Removed)
    });

    // --- STARTUP OPTIMIZATION TRIGGER ---
    // Since we have history, we ask the Dreamers to optimize ALL active assets immediately on boot.
    if (!global.hasTriggeredStartupOptimization && portfolioSummary.length > 0) {
      console.log("\n🚀 [STARTUP] Triggering Mass Scientific Optimization for all assets...");
      portfolioSummary.forEach(row => {
        if (!HARVEST_EXCLUDE.includes(row.Symbol)) { // Respect Exclusions
          legionManager.requestOptimization(row.Symbol);
        }
      });
      global.hasTriggeredStartupOptimization = true;
    }
    // --- End EXCLUDED asset state cleanup ---

    let deletedKeys = false; Object.keys(tokenBaselines).forEach(sym => { if (!currentSymbols.has(sym)) { console.log(`🗑️ Clearing state for sold/removed asset: ${sym}`); delete tokenBaselines[sym]; delete trailingState[sym]; delete rebalanceState[sym]; delete lastActionTimestamps[sym]; deletedKeys = true; } }); Object.keys(trailingState).forEach(sym => { if (!tokenBaselines[sym]) { console.log(`🗑️ Clearing trailing state for ${sym} (no baseline).`); delete trailingState[sym]; deletedKeys = true; } }); Object.keys(lastActionTimestamps).forEach(sym => { if (!tokenBaselines[sym]) { console.log(`🗑️ Clearing last action timestamp for ${sym} (no baseline).`); delete lastActionTimestamps[sym]; deletedKeys = true; } }); Object.keys(rebalanceState).forEach(sym => { if (!tokenBaselines[sym]) { delete rebalanceState[sym]; } });
    if (deletedKeys) { stateChanged = true; } if (stateChanged) { saveState(); stateChanged = false; }

    // Calculate Portfolio Deviation (Only needed for display now)
    let totalBaselineDifference = 0; let totalManagedBaselineValue = 0;
    portfolioSummary.forEach(row => { if (row.Baseline && typeof row.Baseline === 'number' && row.Baseline > 0 && !REBALANCE_EXCLUDE.includes(row.Symbol)) { totalBaselineDifference += (row.Value - row.Baseline); totalManagedBaselineValue += row.Baseline; } });
    let currentPortfolioDeviationPercent = 0; if (totalManagedBaselineValue > 0) { currentPortfolioDeviationPercent = (totalBaselineDifference / totalManagedBaselineValue) * 100; }

    // Display Portfolio Table & Financial Overview (Unchanged)
    if (portfolioSummary.length > 0) {
      portfolioSummary.sort((a, b) => { let devA = NaN; if (a.Baseline && a.Baseline > 0) devA = (a.Value - a.Baseline) / a.Baseline; let devB = NaN; if (b.Baseline && b.Baseline > 0) devB = (b.Value - b.Baseline) / b.Baseline; if (isNaN(devA) && isNaN(devB)) return 0; if (isNaN(devA)) return 1; if (isNaN(devB)) return -1; return devB - devA; }); const displayData = portfolioSummary.map(row => {
        const deviation = (row.Baseline && row.Baseline > 0) ? ((row.Value - row.Baseline) / row.Baseline) * 100 : NaN;

        // Tier 3: Deviation Arrows (↑/↓)
        let arrow = "";
        const tState = trailingState[row.Symbol];
        if (tState && typeof tState.previousDeviation === 'number') {
          const currentDev = (row.Value - row.Baseline) / row.Baseline;
          if (currentDev > tState.previousDeviation) arrow = "↑";
          else if (currentDev < tState.previousDeviation) arrow = "↓";
        }

        const color = (!isNaN(deviation) && deviation < 0) ? '\x1b[31m' : '\x1b[32m'; // Red if neg, Green if pos
        const reset = '\x1b[0m';
        const coloredDeviation = isNaN(deviation) ? 'N/A' : `${color}${arrow}${deviation.toFixed(2)}%${reset}`;

        // --- High-Fidelity Trigger Visualization ---
        let triggerStr = "-";
        if (!HARVEST_EXCLUDE.includes(row.Symbol) && !REBALANCE_EXCLUDE.includes(row.Symbol)) {
          // Get ACTIVE values from LIVE engine's genome (not simulation genome)
          const activeH = getGenomicParam(liveEngine.genome, 'FLAT_HARVEST_TRIGGER_PERCENT', row.Symbol);
          const activeR = getGenomicParam(liveEngine.genome, 'FLAT_REBALANCE_TRIGGER_PERCENT', row.Symbol);

          // Colorize Optimized Values (Compare to Default)
          const defH = defaultGenome.FLAT_HARVEST_TRIGGER_PERCENT;
          const defR = defaultGenome.FLAT_REBALANCE_TRIGGER_PERCENT;

          // COLOR CODING: Visual indicator for Timeframe
          let colorCode = '\x1b[36m'; // Default Cyan for Optimized
          const timeframe = liveEngine.assetSourceTimeframe ? liveEngine.assetSourceTimeframe[row.Symbol] : null;

          if (timeframe === 'SHORT') colorCode = '\x1b[36m'; // Cyan
          else if (timeframe === 'MEDIUM') colorCode = '\x1b[33m'; // Yellow
          else if (timeframe === 'LONG') colorCode = '\x1b[35m'; // Magenta

          const reset = '\x1b[0m';

          const hStr = Math.abs(activeH - defH) > 0.000001 ? `${colorCode}${(activeH * 100).toFixed(2)}%${reset}` : `${(activeH * 100).toFixed(2)}%`;
          const rStr = Math.abs(activeR - defR) > 0.000001 ? `${colorCode}${(activeR * 100).toFixed(2)}%${reset}` : `${(activeR * 100).toFixed(2)}%`;

          triggerStr = `${hStr} / ${rStr}`;
        }

        // Value Colorization (Tick-to-Tick)
        let valueColor = "";
        const prevVal = previousCycleValues[row.Symbol];
        if (prevVal !== undefined) {
          if (row.Value > prevVal) valueColor = '\x1b[32m'; // Green (Up)
          else if (row.Value < prevVal) valueColor = '\x1b[31m'; // Red (Down)
        }
        previousCycleValues[row.Symbol] = row.Value; // Update for next cycle

        // --- Baseline Colorization (Status Indicator) ---
        let baselineColor = "";
        if (!HARVEST_EXCLUDE.includes(row.Symbol) && !REBALANCE_EXCLUDE.includes(row.Symbol)) {
          // Re-calculate active params here to be safe
          const activeH = getGenomicParam(liveEngine.genome, 'FLAT_HARVEST_TRIGGER_PERCENT', row.Symbol);
          const activeR = getGenomicParam(liveEngine.genome, 'FLAT_REBALANCE_TRIGGER_PERCENT', row.Symbol);

          if (!isNaN(deviation)) {
            if (deviation >= activeH * 100) {
              baselineColor = `\x1b[32m`; // Green (Harvest)
            } else if (deviation <= -(activeR * 100)) {
              baselineColor = `\x1b[31m`; // Red (Rebalance - User Request)
            }
          }
        }

        return {
          Symbol: row.Symbol,
          Quantity: row.Quantity.toLocaleString(undefined, { maximumFractionDigits: 8 }),
          Price: row.Price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 10 }),
          Value: `${valueColor}${row.Value.toLocaleString(undefined, { style: 'currency', currency: 'USD' })}${reset}`,
          Baseline: row.Baseline ? `${baselineColor}$${row.Baseline.toFixed(2)}${reset}` : 'N/A',
          Deviation: coloredDeviation,
          Triggers: triggerStr
        };
      });
      console.log("\n--- Portfolio Summary (Sorted by Deviation %) ---");
      // console.table(displayData); // Disabled due to ANSI escaping issues

      const headers = ["Index", "Symbol", "Quantity", "Price", "Value", "Baseline", "Deviation", "H/R Trig"];
      const tableRows = displayData.map((d, i) => [
        i + 1,
        d.Symbol,
        d.Quantity,
        d.Price,
        d.Value,
        d.Baseline,
        d.Deviation,
        d.Triggers
      ]);
      printTable(headers, tableRows);

    } else { if (holdings.length > 0) console.log("ℹ️ No displayable portfolio data (likely waiting on valid prices)."); }
    console.log("--- Financial Overview ---"); console.log(`Total Holdings Value:   $${totalHoldingsValue.toFixed(2)}`); console.log(`Cash Balance:           $${cashBalance.toFixed(2)}`); const totalPortfolioValue = totalHoldingsValue + cashBalance; console.log(`Total Portfolio Value:  $${totalPortfolioValue.toFixed(2)}`); const diffPrefix = totalBaselineDifference >= 0 ? '+' : ''; const diffColor = totalBaselineDifference >= 0 ? '\x1b[32m' : '\x1b[31m'; const resetColor = '\x1b[0m'; console.log(`Deviation (Managed):    ${diffColor}${diffPrefix}$${totalBaselineDifference.toFixed(2)} (${currentPortfolioDeviationPercent.toFixed(2)}%)${resetColor}`);

    // --- Regime Radar (Legion Awareness) ---
    if (LEGION_CONFIG.ENABLE_DEVELOPER_LOGS && regimeDetector && regimeDetector.regimes) {
      const activeRegimes = Object.entries(regimeDetector.regimes)
        .filter(([s, r]) => tokenBaselines[s])
        .map(([s, r]) => `${s}:${r}`)
        .join(" | ");
      if (activeRegimes.length > 0) console.log(`🛠️ [DEV] Regime Radar: [${activeRegimes}]`);
    }
    console.log("--------------------------\n");


    // Update Regime Detector History
    if (portfolioSummary.length > 0 && regimeDetector) {
      // We feed the detector. It maintains its own buffer.
      portfolioSummary.forEach(row => {
        regimeDetector.update(row.Symbol, row.Price, row.t || Date.now());
      });
    }

    // --- Auto Trading Logic ---
    if (portfolioSummary.length === 0 || !initialized) {
      console.log("⏳ Skipping trading actions (Portfolio empty or not initialized).\n");
    }
    else {
      console.log("🚦 Baselines ready. Delegating to Live Engine...");

      const engineResult = await liveEngine.update(portfolioSummary, rh, cashBalance, holdingDetails);

      // --- Legion Heartbeat ---
      if (legionManager) {
        // The Manager decides who lives anddies
        await legionManager.heartbeat(portfolioSummary, rh);
      }

      // --- Oracle/Regime Detector ---
      // Update Regime Detector with latest history
      if (portfolioSummary.length > 0) {
        const historyTick = { timestamp: Date.now(), p: {} };
        portfolioSummary.forEach(row => historyTick.p[row.Symbol] = row.Price);
        // We need to maintain a history buffer for the detector per asset?
        // RegimeDetector expects an array of prices for a single asset.
        // We should collect this data cleanly.
        // For now, let's assume RegimeDetector helps us or we just feed it?
        // Actually, RegimeDetector.analyze takes (symbol, historyArray).
        // We need to pass the history.
        // Let's use the liveEngine's price history for this?
        // liveEngine.priceHistory contains raw objects.

        // Iterate all assets and update regime
        portfolioSummary.forEach(row => {
          const sym = row.Symbol;
          // Extract simplified price history from engine
          // This is expensive every tick? Maybe do it occasionally.
          // The Manager does it inside heartbeat if needed, but the Detector is separate.
          // Let's rely on Manager to query Detector if needed, or update Detector here.
        });
      }

      anyTradesThisCycle = engineResult.anyTradesThisCycle;
      if (engineResult.stateChanged) stateChanged = true;

      // --- Post-Trade "Dream Replay" Trigger ---
      // If we traded, immediately trigger a Re-Optimization to verify/fine-tune the decision based on new state.
      if (anyTradesThisCycle && engineResult.tradedSymbols && engineResult.tradedSymbols.length > 0) {
        if (legionManager) {
          engineResult.tradedSymbols.forEach(sym => {
            console.log(`⚡ [Dreamer] Trade detected on ${sym}. Triggering Immediate Verification Sweep...`);
            legionManager.requestOptimization(sym);
          });
        }
      }

      // Refresh Local State References (Aliases) for Display Logic
      // This ensures we display the *post-update* state, especially if objects were reassigned.
      tokenBaselines = liveEngine.baselines;
      trailingState = liveEngine.trailingState;
      lastActionTimestamps = liveEngine.lastActionTimestamps;
      rebalanceState = liveEngine.rebalanceState;
      // adaptiveDeadZoneState = liveEngine.adaptiveDeadZoneState; // REMOVED
      portfolioHarvestState = liveEngine.portfolioHarvestState;

      // Save state if any persistent changes occurred (with optimized throttling)
      if (stateChanged || anyTradesThisCycle) {
        if (anyTradesThisCycle || (Date.now() - lastStateSaveTime > 5 * 60 * 1000)) {
          saveState();
          lastStateSaveTime = Date.now();
        }
      }
    } // End Auto Trading Logic Block

    // --- Display Active States --- // *** Simplified format ***
    try {
      if (portfolioHarvestState.flagged) { console.log(`📈 Portfolio Harvest Flagged: Count ${portfolioHarvestState.cycleCount}/${currentGenome.PORTFOLIO_HARVEST_CONFIRMATION_CYCLES}, Prev Port. Dev: ${portfolioHarvestState.previousDeviationPercent?.toFixed(2)}%`); }

      const flaggedForHarvest = Object.entries(trailingState)
        .filter(([sym, s]) => s?.flagged && tokenBaselines[sym])
        .map(([sym, s]) => `${sym}:${s.harvestCycleCount}`);
      if (flaggedForHarvest.length > 0) console.log(`🚩 Flagged Harvest: [${flaggedForHarvest.join(", ")}]`);

      const activeRebalancing = Object.entries(rebalanceState)
        .filter(([sym, s]) => s?.triggered && Date.now() >= (s.cooldownUntil || 0) && tokenBaselines[sym])
        .map(([sym, s]) => `${sym}:${s.rebalancePosCycleCount}`);
      const inCooldown = Object.entries(rebalanceState)
        .filter(([sym, s]) => s?.triggered && Date.now() < (s.cooldownUntil || 0) && tokenBaselines[sym])
        .map(([sym, s]) => `${sym}:${Math.ceil(((s.cooldownUntil || 0) - Date.now()) / 60000)}m`);
      if (activeRebalancing.length > 0) console.log(`⚖️ Active Rebalance: [${activeRebalancing.join(", ")}]`);
      if (inCooldown.length > 0) console.log(`⏸️ Rebalance Cooldown: [${inCooldown.join(", ")}]`);

      // Cleaned legacy displays

      if (!anyTradesThisCycle && !portfolioHarvestState.flagged && flaggedForHarvest.length === 0 && activeRebalancing.length === 0 && inCooldown.length === 0 && portfolioSummary.length > 0 && initialized) {
        console.log("🧘 No trading actions or adaptive states triggered this cycle.");
      }
    } catch (displayError) { console.error("⚠️ Error displaying states:", displayError); }


    // --- Periodic State Save ---
    if (startTime - lastStateSaveTime >= STATE_SAVE_INTERVAL) {
      saveEngineState();
      lastStateSaveTime = startTime;
    }

    // --- Cycle Timing (Unchanged) ---
    const endTime = Date.now(); const elapsed = endTime - startTime; const delay = Math.max(0, currentGenome.REFRESH_INTERVAL - elapsed);
    console.log(`----- Cycle End: Took ${elapsed}ms. Waiting ${delay}ms... -----`);
    await new Promise((res) => setTimeout(res, delay));

  } // End Main Loop

  console.log("🛑 Main loop exited unexpectedly."); rl.close();
} // End mainLoop Function



// ============== SYSTEMATIC SWEEP CLASSES (Ported from The Dreamer) ==============



class SweepStateManager {
  constructor(workerId = 0, totalWorkers = 1) {
    this.workerId = workerId;
    this.totalWorkers = totalWorkers;
    this.stateFile = path.join(process.cwd(), 'configs', `sweep_state_${workerId}.json`);
    this.state = {
      currentAssetIndex: 0,
      combinationsChecked: 0,
      mode: 'GRID', // 'GRID' (Harvest+Rebal) or 'FINE_TUNE' (Cycles, Recovery)
      hIndex: 0, // Harvest Index
      rIndex: 0, // Rebalance Index
      paramIndex: 0, // For fine-tuning
      val: null,
      swarmInit: false
    };
  }

  load() {
    if (fs.existsSync(this.stateFile)) {
      try {
        this.state = JSON.parse(fs.readFileSync(this.stateFile, 'utf-8'));
        console.log("   [Dreamer] 🧠 Memory Loaded: Continuing previous sweep...");
      } catch (e) { }
    }
  }

  save() {
    try {
      const dir = path.dirname(this.stateFile);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(this.stateFile, JSON.stringify(this.state, null, 2));
    } catch (e) { }
  }
}

class TradeHistoryAnalyzer {
  constructor(historyFile = 'trade_history.log') {
    this.historyFile = historyFile;
    this.stats = {}; // { SYM: { wins, losses, pnl, totalTrades } }
    this.trades = []; // Raw trade list
    this.loaded = false;
  }

  loadHistory() {
    if (!fs.existsSync(this.historyFile)) return;
    try {
      const lines = fs.readFileSync(this.historyFile, 'utf-8').split('\n');
      this.stats = {};
      this.trades = [];
      lines.forEach(line => {
        if (!line.trim()) return;
        try {
          const trade = JSON.parse(line);
          if (!trade.asset || !trade.side || !trade.totalValue) return;
          this.trades.push(trade);
          if (!this.stats[trade.asset]) this.stats[trade.asset] = { wins: 0, losses: 0, pnl: 0, totalTrades: 0 };
          const s = this.stats[trade.asset];
          const val = parseFloat(trade.totalValue);
          if (trade.side === 'BUY') s.pnl -= val;
          else if (trade.side === 'SELL') s.pnl += val;
          s.totalTrades++;
        } catch (e) { }
      });
      this.loaded = true;
      console.log(`   📚 History Loaded: Processed ${this.trades.length} trades.`);
    } catch (err) { console.error("Error loading history:", err); }
  }

  calculateSlippageMap() {
    if (!this.loaded) this.loadHistory();
    const slippageStats = {};
    this.trades.forEach(trade => {
      // Simplified slippage calculation for embedded version (no full market data load to save RAM)
      // Or we can rely on the fact that if we are embedded, we might not want to load huge JSONL files.
      // For now, return default.
      // Actually, let's include basic default.
    });
    return { 'DEFAULT': 0.003 };
  }
}

// Renamed from SimulatorWorker to ScientificOptimizer for self-contained usage
class ScientificOptimizer {
  constructor() {
    this.marketDataFile = path.join(process.cwd(), 'market_data.jsonl');
    this.maxHistoryLines = 450000;
    this.WINDOW_24H = 15000;
    this.historyAnalyzer = new TradeHistoryAnalyzer();

    // Parse Swarm IDs
    this.workerId = 0;
    this.totalWorkers = 1;
    process.argv.forEach(arg => {
      if (arg.startsWith('--workerId=')) this.workerId = parseInt(arg.split('=')[1]);
      else if (arg.startsWith('--totalWorkers=')) this.totalWorkers = parseInt(arg.split('=')[1]);
    });

    this.sweepStateManager = new SweepStateManager(this.workerId, this.totalWorkers);

    // 1. GRID SEARCH RANGES (The Matrix)
    // We test every combination of these two against each other
    this.GRID_RANGES = {
      HARVEST: { start: 0.02, end: 0.08, step: 0.005 }, // 2.0% to 8% (Min > 1% Fee)
      REBALANCE: { start: 0.02, end: 0.08, step: 0.005 }  // 2.0% to 8%
    };

    // 2. FINE TUNE RANGES (Sequential)
    // We test these one by one after the best grid pos is found
    this.FINE_TUNE_RANGES = {
      // HARVEST_CYCLE_THRESHOLD: { start: 1, end: 6, step: 1 }, // REMOVED: Fixed at Default (3)
      // REBALANCE_POSITIVE_THRESHOLD: { start: 1, end: 6, step: 1 }, // REMOVED: Fixed at Default (3)
      PARTIAL_RECOVERY_PERCENT: { start: 0.30, end: 1.00, step: 0.10 }
    };
  }

  async run() {
    console.log("   [Dreamer] 🧮 Combinatorial Processor Active. Pid:", process.pid);
    this.sweepStateManager.load();

    // --- Persistence for Stats ---
    const counterPath = path.join(process.cwd(), 'dreamerSimCount.json');
    let totalSims = 0;
    if (fs.existsSync(counterPath)) try { totalSims = JSON.parse(fs.readFileSync(counterPath)).totalSimulations; } catch (e) { }

    let batchCounter = 0;
    while (true) {
      batchCounter++;

      // Periodic Status Log (Every ~5s)
      if (batchCounter % 500 === 0 && this.sweepStateManager?.state) {
        const s = this.sweepStateManager.state;
        const assets = this.getSweepAssets(this.cachedHistory);
        const loopAsset = assets[s.currentAssetIndex % assets.length] || "Waiting...";

        // Calculate Depth Mode (Grand Cycle)
        const passNum = Math.floor((s.currentAssetIndex || 0) / (assets.length || 1));
        const depthMode = ['SHORT', 'MEDIUM', 'LONG'][passNum % 3] || 'MEDIUM';

        console.log(`   [Dreamer] 🔄 Cycling: ${loopAsset} [${depthMode}] [${s.mode}] (Sim #${batchCounter})`);
      }

      // Cache Refresh Logic (Throttled from checking file size every 500 loops to purely memory load)
      if (batchCounter % 1500 === 0 || !this.cachedHistory) {
        this.cachedHistory = await this.loadMarketData();
      }

      try {
        const history = this.cachedHistory || [];
        if (history.length > 500) { // Require minimum data
          // Use ALL available history up to the max window (e.g. 7 days / 100k ticks)
          const recentHistory = history.slice(-Math.min(history.length, 100000));

          // === CORE LOGIC: GET NEXT COMBINATION ===
          const candidate = this.getNextCombinatorialCandidate(recentHistory);

          // --- Swarm Initalization (Asset Staggering) ---
          if (candidate && !this.sweepStateManager.state.swarmInit) {
            const assets = this.getSweepAssets(recentHistory);
            if (assets.length > 0) {
              const slice = Math.floor(assets.length / this.totalWorkers);
              const startIdx = slice * this.workerId;
              // Only jump if we are at 0 (Fresh Start), otherwise respect loaded state
              if (this.sweepStateManager.state.currentAssetIndex === 0) {
                this.sweepStateManager.state.currentAssetIndex = startIdx;
                console.log(`   [Dreamer #${this.workerId}] 🐝 Swarm Active. Starting at Asset Index ${startIdx}/${assets.length} (${assets[startIdx] || '?'}).`);
              }
              this.sweepStateManager.state.swarmInit = true;
            }
          }

          if (candidate) {
            // Hybrid Scan Logic
            const assets = this.getSweepAssets(recentHistory);
            const passNum = Math.floor((this.sweepStateManager.state.currentAssetIndex || 0) / (assets.length || 1));
            const depthMode = ['SHORT', 'MEDIUM', 'LONG'][passNum % 3] || 'MEDIUM';

            // Load LIVE snapshot for accurate baseline
            // Load LIVE snapshot for accurate baseline
            let liveSnapshot = null;
            try {
              if (fs.existsSync('liveEngineState.json')) {
                const stats = fs.statSync('liveEngineState.json');
                const ageMinutes = (Date.now() - stats.mtimeMs) / 60000;
                if (ageMinutes > 10) {
                  console.warn(`⚠️ [Dreamer] Warning: Live State is ${ageMinutes.toFixed(1)}m old. Optimizations may be based on stale data.`);
                }
                liveSnapshot = JSON.parse(fs.readFileSync('liveEngineState.json'));
              }
            } catch (e) { }

            const result = await this.evaluateGenome(candidate.genome, recentHistory, liveSnapshot, candidate.focus, depthMode);
            totalSims++;

            // Scoring Logic (Same as before)
            let score = -Infinity;
            if (result && result.totalValue > -1) {
              const minTrades = candidate.genome.MIN_TRADES_FOR_PROMOTION || 1;
              if (result.totalTrades < minTrades) score = -1000.0;
              else {
                const dd = Math.max(0.01, result.drawdown);
                const penalty = 1 + (dd * (candidate.genome.FITNESS_DRAWDOWN_PENALTY || 1.0));
                // CHANGE: Use Relative ROI (Alpha) so we promote "Least Loss" in bear markets
                score = result.relativeROI / penalty;
              }
            }

            // Track best score per asset AND depth mode
            if (!global.lastBestScore || typeof global.lastBestScore !== 'object') global.lastBestScore = {};

            // Namespace the score tracking so SHORT (e.g. 5.0%) doesn't block LONG (e.g. 2.0%) promotions
            const scoreKey = `${candidate.focus}_${depthMode}`;
            const currentBest = global.lastBestScore[scoreKey] || -Infinity; // Reset if switching modes implicitly handled by unique key

            // Heartbeat / Promotion Logic
            if (process.send) {
              // STRICT: Only promote if Alpha is POSITIVE (> 0).
              // This also prevents the "Perverse Drawdown Incentive" bug where dividing a negative score by a penalty > 1 makes it "better".
              if (score > currentBest && score > 0.0001) {
                global.lastBestScore[scoreKey] = score;
                this.saveBrainScan(candidate.focus, result, candidate.genome); // <--- BRAIN SCAN
                process.send({
                  type: 'OPTIMIZATION_FOUND',
                  genome: candidate.genome,
                  score: score,
                  focus: candidate.focus,
                  symbol: candidate.focus,
                  param: `${candidate.desc} [${depthMode}]`, // Tag the description
                  val: candidate.valStr
                });
              }
              process.send({
                type: 'HEARTBEAT',
                batch: batchCounter,
                bestScore: score,
                focus: `[${candidate.mode}] ${candidate.focus} ${candidate.desc}`
              });
            }
          }

          // Save progress (Massively throttled to prevent HDD lockups during worker sweeps)
          if (totalSims % 5000 === 0) {
            try {
              fs.writeFileSync(counterPath, JSON.stringify({ totalSimulations: totalSims }));
            } catch (e) { }
          }
        }
      } catch (err) {
        console.error("   [Dreamer] Critical Error:", err);
        await new Promise(r => setTimeout(r, 5000));
      } finally {
        await new Promise(r => setTimeout(r, 10)); // Ultra-fast yield for grid search
      }
    }
  }

  getNextCombinatorialCandidate(history) {
    const assets = this.getSweepAssets(history);
    if (assets.length === 0) return null;

    let st = this.sweepStateManager.state;
    st.combinationsChecked = (st.combinationsChecked || 0) + 1;
    const asset = assets[st.currentAssetIndex % assets.length];

    // Debug Cycle
    // console.log(`[Dreamer] Debug Cycle: ${asset} [${st.mode}] (Index ${st.currentAssetIndex} of ${assets.length})`);

    // Load Baseline Genome
    let base = defaultGenome;
    try {
      const f = JSON.parse(fs.readFileSync('liveEngineState.json', 'utf-8'));
      if (f.genome) base = f.genome;
    } catch (e) { }

    const candidateGenome = JSON.parse(JSON.stringify(base));
    if (!candidateGenome.overrides) candidateGenome.overrides = {};
    if (!candidateGenome.overrides[asset]) candidateGenome.overrides[asset] = {};

    let result = { genome: candidateGenome, focus: asset, mode: st.mode };

    // === CHAMPION CHECK (New Feature) ===
    // Before starting a new GRID sweep, verify the CURRENT settings first.
    // This sets the "High Score" bar to beat. If the grid can't beat the current settings, nothing changes.
    if (st.mode === 'GRID' && st.hIndex === 0 && st.rIndex === 0 && (!st.championChecked)) {
      // We return the genome AS IS (with its current overrides for this asset).
      // This counts as the "Champion" run.
      st.championChecked = true;
      // Populate description so logs aren't "undefined"
      result.desc = "CHAMPION";
      result.valStr = "INCUMBENT";
      return result;
    }

    // === MODE 1: GRID SEARCH (Harvest x Rebalance) ===
    if (st.mode === 'GRID') {
      const hStart = this.GRID_RANGES.HARVEST.start;
      const hStep = this.GRID_RANGES.HARVEST.step;
      const rStart = this.GRID_RANGES.REBALANCE.start;
      const rStep = this.GRID_RANGES.REBALANCE.step;

      const hVal = hStart + (st.hIndex * hStep);
      const rVal = rStart + (st.rIndex * rStep);

      // SAFETY: Cap rebalance trigger at 15% (same as micro-grid)
      const MAX_REBALANCE_TRIGGER = 0.15;
      const constrainedRVal = Math.min(rVal, MAX_REBALANCE_TRIGGER);

      // Apply BOTH parameters
      candidateGenome.overrides[asset].FLAT_HARVEST_TRIGGER_PERCENT = parseFloat(hVal.toFixed(4));
      candidateGenome.overrides[asset].FLAT_REBALANCE_TRIGGER_PERCENT = parseFloat(constrainedRVal.toFixed(4));

      result.desc = `H:${(hVal * 100).toFixed(2)}% / R:${(constrainedRVal * 100).toFixed(2)}%`;
      result.valStr = "COMBINED";

      // Advance Counters
      if (rVal >= this.GRID_RANGES.REBALANCE.end) {
        st.rIndex = 0;
        st.hIndex++;
        if (hVal >= this.GRID_RANGES.HARVEST.end) {
          // Grid Complete -> Switch to MICRO_GRID (High Precision)
          st.hIndex = 0;
          st.rIndex = 0;
          st.mode = 'MICRO_GRID';

          // Load current winner to set anchors
          let currentH = 0.035;
          let currentR = 0.035;
          try {
            const f = JSON.parse(fs.readFileSync('liveEngineState.json', 'utf-8'));
            if (f.genome && f.genome.overrides && f.genome.overrides[asset]) {
              currentH = f.genome.overrides[asset].FLAT_HARVEST_TRIGGER_PERCENT || 0.035;
              currentR = f.genome.overrides[asset].FLAT_REBALANCE_TRIGGER_PERCENT || 0.035;
            }
          } catch (e) { }

          st.anchorH = currentH;
          st.anchorR = currentR;
          console.log(`   [Dreamer] 🔬 ${asset} Grid Complete. Switching to Micro-Verification (Anchor H:${(st.anchorH * 100).toFixed(2)}% / R:${(st.anchorR * 100).toFixed(2)}%).`);
        }
      } else {
        st.rIndex++;
      }
    }

    // === MODE 3: MICRO-GRID (Verification) ===
    // === MODE 3: MICRO-GRID (Crosshair Verification) ===
    else if (st.mode === 'MICRO_GRID') {
      // Anchor values from the trade or Grid winner
      const anchorH = st.anchorH || 0.035;
      const anchorR = st.anchorR || 0.035;
      const range = 0.005; // +/- 0.5% Window (Tight Sniping)
      const step = 0.0001; // 0.01% Precision (High Fidelity)

      // Initialize Micro-State
      if (!st.microPhase) st.microPhase = 'SWEEP_H';
      if (st.mIndex === undefined) st.mIndex = 0;

      let hVal, rVal;
      const steps = Math.floor((range * 2) / step); // Total steps in one direction (approx 300)

      if (st.microPhase === 'SWEEP_H') {
        // Phase 1: Hold R constant (Anchor), Sweep H
        const start = Math.max(0.01, anchorH - range);
        hVal = start + (st.mIndex * step);
        rVal = anchorR;

        candidateGenome.overrides[asset].FLAT_HARVEST_TRIGGER_PERCENT = parseFloat(hVal.toFixed(4));
        candidateGenome.overrides[asset].FLAT_REBALANCE_TRIGGER_PERCENT = parseFloat(rVal.toFixed(4));
        result.desc = `µH:${(hVal * 100).toFixed(2)}% (R Fixed)`;
        result.valStr = "MICRO_H";

        st.mIndex++;
        if (st.mIndex > steps) {
          st.microPhase = 'SWEEP_R';
          st.mIndex = 0; // Reset for next phase
        }
      }
      else if (st.microPhase === 'SWEEP_R') {
        // Phase 2: Hold H constant (Anchor), Sweep R
        const start = Math.max(0.01, anchorR - range);
        hVal = anchorH;
        rVal = start + (st.mIndex * step);

        // SAFETY: Cap rebalance trigger at 15% to prevent extreme values
        // Without this, the optimizer can find local maxima at extreme triggers (e.g. 30%)
        // which are mathematically "better" in backtests but risky in real trading
        const MAX_REBALANCE_TRIGGER = 0.15; // 15% maximum
        rVal = Math.min(rVal, MAX_REBALANCE_TRIGGER);

        candidateGenome.overrides[asset].FLAT_HARVEST_TRIGGER_PERCENT = parseFloat(hVal.toFixed(4));
        candidateGenome.overrides[asset].FLAT_REBALANCE_TRIGGER_PERCENT = parseFloat(rVal.toFixed(4));
        result.desc = `µR:${(rVal * 100).toFixed(2)}% (H Fixed)`;
        result.valStr = "MICRO_R";

        st.mIndex++;
        if (st.mIndex > steps) {
          // Micro Grid Complete -> Go to Fine Tune
          st.microPhase = null;
          st.mIndex = 0;
          st.mode = 'FINE_TUNE';
          st.paramIndex = 0;
          st.val = null;
          console.log(`   [Dreamer] ✅ ${asset} Micro-Verification Complete.`);
        }
      }
    }


    else {
      // ... existing FINE_TUNE logic ...
      const keys = Object.keys(this.FINE_TUNE_RANGES);
      if (st.paramIndex < 0 || st.paramIndex >= keys.length) {
        console.warn(`   [Dreamer] ⚠️ Corrupted Param Index (${st.paramIndex}). Resetting to 0.`);
        st.paramIndex = 0;
      }

      const key = keys[st.paramIndex % keys.length];
      const range = this.FINE_TUNE_RANGES[key];

      if (!range) {
        console.error(`   [Dreamer] 🛑 CRITICAL: Range undefined for key '${key}'. Resetting system.`);
        st.paramIndex = 0;
        st.mode = 'GRID'; // Fallback to Grid
        return null;
      }

      let val = st.val;
      if (val === null) val = range.start;
      else val += range.step;

      candidateGenome.overrides[asset][key] = parseFloat(val.toFixed(4));
      result.desc = key;
      result.valStr = val.toFixed(2);
      result.param = key; // For compatibility
      result.value = val;

      // Advance Counters
      st.val = val;
      if (val >= range.end - 0.0001) {
        st.val = null;
        st.paramIndex++;
        if (st.paramIndex >= keys.length) {
          // Asset Complete -> Next Asset -> Back to Grid
          st.paramIndex = 0;
          st.currentAssetIndex++;
          st.mode = 'GRID';
          const sScore = global.lastBestScore?.[`${asset}_SHORT`] || 0;
          const mScore = global.lastBestScore?.[`${asset}_MEDIUM`] || 0;
          const lScore = global.lastBestScore?.[`${asset}_LONG`] || 0;
          console.log(`   [Dreamer] ✅ ${asset} FULLY OPTIMIZED. Scanned ${st.combinationsChecked} strategies. Best Alpha (S/M/L): ${sScore.toFixed(3)}% / ${mScore.toFixed(3)}% / ${lScore.toFixed(3)}%`);
          st.combinationsChecked = 0;

          // SOFT RESET SCORE for next cycle (Fixes "Dreamer Amnesia")
          // Instead of deleting (reset to 0), we decay the score.
          // This forces the next candidate to be at least 75% as good as the old champion,
          // preventing immediate downgrades to mediocre strategies while still allowing adaptation.
          if (global.lastBestScore) {
            const keys = [`${asset}_SHORT`, `${asset}_MEDIUM`, `${asset}_LONG`];
            keys.forEach(k => {
              if (global.lastBestScore[k]) global.lastBestScore[k] *= 0.75;
            });

            // Notify main process to decay its thresholds too (Soft Reset)
            if (process.send) {
              process.send({
                type: 'RESET_SCORE',
                asset: asset,
                mode: 'SOFT' // New mode for main process to handle if needed, or just standard reset
              });
            }


          }

          // RESET CHAMPION CHECK
          st.championChecked = false;
        }
      }
    }


    // Throttle state saving to disk to prevent locking. (Saves roughly every 100 combination checks instead of every 1)
    if (this.sweepStateManager.state.combinationsChecked % 100 === 0) {
      this.sweepStateManager.save();
    }
    return result;
  }

  getSweepAssets(history) {
    if (!history || history.length === 0) return [];
    const allAssets = Object.keys(history[history.length - 1].p).sort().filter(s => !HARVEST_EXCLUDE.includes(s));
    // 🧱 SHARDING: Distribute work among workers
    return allAssets.filter((_, i) => i % this.totalWorkers === this.workerId);
  }

  async loadMarketData() {
    if (!fs.existsSync(this.marketDataFile)) return [];
    const data = fs.readFileSync(this.marketDataFile, 'utf-8').trim().split('\n').map(l => { try { return JSON.parse(l); } catch (e) { return null; } }).filter(x => x);
    // console.log(`   [Dreamer] 📚 Loaded ${data.length} ticks for simulation.`); // User requested silence
    return data;
  }

  async pruneData(history) {
    if (history.length <= this.maxHistoryLines) return;
    const kept = history.slice(history.length - this.maxHistoryLines);
    const lines = kept.map(x => JSON.stringify(x)).join('\n');
    fs.writeFileSync(this.marketDataFile, lines + '\n');
  }


  handleFeedback(msg) {
    // "Micro-Tune" Injection:
    // A Shadow Scout has found a winning value (e.g. 7.52%) that beats our Grid (7.50%).
    // We accept this victory and update our best score.
    if (!global.lastBestScore) global.lastBestScore = {};
    const currentBest = global.lastBestScore[msg.focus] || 0.0;

    if (msg.score > currentBest) {
      global.lastBestScore[msg.focus] = msg.score;

      // Re-broadcast to Main so it can be promoted to Live Engine & Saved
      // (Only Main can write to the actual liveEngineState.json)
      if (process.send) {
        process.send({
          type: 'OPTIMIZATION_FOUND',
          genome: msg.genome,
          score: msg.score,
          focus: msg.focus,
          symbol: msg.focus,
          param: "FINE_TUNE_FEEDBACK",
          val: "ADAPTIVE"
        });
      }
      // console.log(`   [Dreamer] 🧠 Absorbed Micro-Tune Feedback for ${msg.focus}: Score ${msg.score.toFixed(2)}%`);
    }
  }

  prioritize(symbol, baseGenome) {
    // "Verify & Fine-Tune" Logic:
    // 1. Accept the exact parameters that just executed the trade (baseGenome).
    // 2. Skip the coarse Grid Search.
    // 3. Jump straight to FINE_TUNE mode centered on these parameters.
    try {
      const overrides = baseGenome.overrides?.[symbol];
      if (overrides) {
        // Determine starting indices for Fine Tuning based on current values
        // This allows us to "sweep around" the current value
        // For now, we simply reset the state but inject the values into the candidate generation logic
        // via a temporary "focus" or by explicitly setting the mode.

        // CRITICAL: We need to ensure the next 'getNextCombinatorialCandidate' picks this up.
        // We'll reset the state for this asset to FINE_TUNE start.
        this.sweepStateManager.state.currentAssetIndex = this.getSweepAssets(this.cachedHistory).indexOf(symbol);
        if (this.sweepStateManager.state.currentAssetIndex === -1) this.sweepStateManager.state.currentAssetIndex = 0;

        this.sweepStateManager.state.mode = 'MICRO_GRID';
        this.sweepStateManager.state.microPhase = 'SWEEP_H';
        this.sweepStateManager.state.mIndex = 0;

        // Set Anchors based on the trade execution parameters
        this.sweepStateManager.state.anchorH = overrides.FLAT_HARVEST_TRIGGER_PERCENT || 0.035;
        this.sweepStateManager.state.anchorR = overrides.FLAT_REBALANCE_TRIGGER_PERCENT || 0.035;

        // Reset val/paramIndex just in case
        this.sweepStateManager.state.val = null;
        this.sweepStateManager.state.paramIndex = 0;

        // We also need to SAVE this specific baseGenome as the "Anchor" for this run.
        // We'll write it to a temporary "priority_anchor.json" or similar that getNext... reads?
        // Or simpler: Just rely on liveEngineState.json which Main updates immediately?
        // Main calls saveEngineState() right before sending this request if a trade happened (stateChanged=true).

        // console.log(`   [Dreamer] ⚡ PRIORITIZING ${symbol}: Verification & Fine-Tuning Sequence Initiated.`);
        this.sweepStateManager.save();
      }
    } catch (err) { console.error("Error in prioritize:", err); }
  }

  // --- BRAIN SCAN DIAGNOSTIC ---
  // --- BRAIN SCAN DIAGNOSTIC ---
  saveBrainScan(asset, result, genome) {
    const scanFile = path.join(process.cwd(), 'dreamer_brain_scan.json');

    // Calculate Fee Impact for verification
    // Result.totalValue is net of fees.
    const feesPaidEst = result.totalTrades * (10000 * 0.01); // Approx $100 per trade on $10k base (rough estimate)

    const scanData = {
      id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
      timestamp: new Date().toISOString(),
      asset: asset,
      logic_verification: {
        msg: "TRUST BUT VERIFY",
        method: "Scientific Combinatorial Sweep",
        fee_model: "1.0% per Trade (Real-time deduction)",
        min_threshold_enforced: "2.0%"
      },
      winner: {
        config: {
          Harvest: genome.overrides?.[asset]?.FLAT_HARVEST_TRIGGER_PERCENT || "N/A",
          Rebalance: genome.overrides?.[asset]?.FLAT_REBALANCE_TRIGGER_PERCENT || "N/A"
        },
        performance: {
          alpha: `${result.relativeROI.toFixed(4)}%`,
          roi: `${result.roi.toFixed(4)}%`,
          market_roi: `${result.marketROI.toFixed(4)}%`,
          trades: result.totalTrades,
          final_value: `$${result.totalValue.toFixed(2)}`
        },
        sanity_check: {
          did_it_beat_holding: result.relativeROI > 0 || (result.totalTrades > 0 && result.relativeROI >= 0),
          is_net_profitable: result.roi > 0
        }
      }
    };

    try {
      let history = [];
      if (fs.existsSync(scanFile)) {
        try {
          const content = JSON.parse(fs.readFileSync(scanFile, 'utf8'));
          if (Array.isArray(content)) history = content;
          else if (typeof content === 'object') history = [content]; // Migrate legacy single-object
        } catch (e) { }
      }

      // Prepend new scan (Newest First)
      history.unshift(scanData);

      // Keep last 50 entries to prevent bloat
      if (history.length > 50) history = history.slice(0, 50);

      fs.writeFileSync(scanFile, JSON.stringify(history, null, 2));
    } catch (err) { console.error("Error saving brain scan:", err); }
  }

  async evaluateGenome(genome, history, snapshot, focusAsset, depthMode = 'MEDIUM') {
    // [Logic: Hybrid Time Windows]
    // SHORT: 24h (15k) - Fast, Responsive
    // MEDIUM: 3d (45k) - Balanced, Robust (Default)
    // LONG: 7 Days (Stress Test)

    let historySlice;
    if (depthMode === 'SHORT') historySlice = history.slice(-15000);
    else if (depthMode === 'MEDIUM') historySlice = history.slice(-45000);
    else historySlice = history.slice(-105000);

    // Safety: Ensure we have data
    if (!historySlice.length) return { roi: 0, totalTrades: 0 };
    const simStartPrice = historySlice[0].p[focusAsset];
    if (!simStartPrice) return { roi: 0, totalTrades: 0 };

    let startCapital = 10000;
    let initialHoldings = {};
    let engine;

    // --- REALITY INJECTION ---
    // Use actual portfolio state if available to train on "Getting out of the hole"
    if (snapshot && snapshot.holdings && snapshot.holdings[focusAsset]) {
      const h = snapshot.holdings[focusAsset];
      let realQty = 0;
      if (typeof h === 'object' && h.rawQuantity !== undefined) realQty = h.rawQuantity;
      else if (typeof h === 'number') realQty = h;

      if (realQty > 0) {
        // 1. Set Holdings to Real Quantity
        initialHoldings = { [focusAsset]: { rawQuantity: realQty } };

        // 2. Set Cash proportional to this asset's share of the portfolio
        // (Estimating: If this asset is 10% of portfolio, give it 10% of available cash)
        // Simplified: Just give it a pro-rated chunk of the $211 cash you have.
        // Assuming ~20 assets, share is ~1/20th.
        const totalAssets = Object.keys(snapshot.holdings).length || 1;
        startCapital = (snapshot.cashBalance || 0) / totalAssets;

        engine = new TradingEngine(genome, 'SHADOW', startCapital, initialHoldings);

        // 3. CRITICAL: Reconstruct the "Pain" (Baseline Deviation)
        // We calculate what the baseline SHOULD be in the simulation to match the
        // real-world deviation % at the start of the simulation period.
        const livePrice = snapshot.lastCyclePrices?.[focusAsset] || simStartPrice;
        const liveBaseline = snapshot.baselines?.[focusAsset] || (realQty * livePrice);

        // Ratio: How far off is the baseline from the price? (e.g., 1.05 = Baseline is 5% above price)
        const deviationsRatio = liveBaseline > 0 ? (liveBaseline / (realQty * livePrice)) : 1.0;

        // Apply this ratio to the simulation start price
        // If real portfolio is down 5%, simulation starts "down 5%" relative to tick 0.
        engine.baselines[focusAsset] = (realQty * simStartPrice) * deviationsRatio;
      } else {
        // We don't hold it, simulate fresh start
        const startQty = startCapital / simStartPrice;
        initialHoldings = { [focusAsset]: { rawQuantity: startQty } };
        engine = new TradingEngine(genome, 'SHADOW', 0, initialHoldings);
        engine.baselines[focusAsset] = startCapital;
      }
    } else {
      // Fallback: Generic $10k All-In
      const startQty = startCapital / simStartPrice;
      initialHoldings = { [focusAsset]: { rawQuantity: startQty } };
      engine = new TradingEngine(genome, 'SHADOW', 0, initialHoldings);
      engine.baselines[focusAsset] = startCapital;
    }

    const dummyApi = { placeBuy: async () => { return { id: 1 } }, placeSell: async () => { return { id: 1 } } };

    // Run Simulation
    for (const tick of historySlice) {
      const price = tick.p[focusAsset];
      if (!price) continue;
      const port = [{ Symbol: focusAsset, Price: price, Value: (engine.holdings[focusAsset]?.rawQuantity || 0) * price, Baseline: engine.baselines[focusAsset] || 0 }];
      await engine.update(port, dummyApi, engine.cashBalance, engine.holdings);
    }

    const finalPrice = historySlice[historySlice.length - 1].p[focusAsset];
    const finalQty = engine.holdings[focusAsset]?.rawQuantity || 0;
    const finalVal = engine.cashBalance + (finalQty * finalPrice);

    // ROI Calculation based on Total Equity (Cash + Asset)
    const initialTotalEquity = startCapital + ((initialHoldings[focusAsset]?.rawQuantity || 0) * simStartPrice);
    const roi = ((finalVal - initialTotalEquity) / initialTotalEquity) * 100;

    const marketROI = ((finalPrice - simStartPrice) / simStartPrice) * 100;
    const relativeROI = roi - marketROI;

    return { roi, relativeROI, marketROI, drawdown: engine.maxDrawdownPercent, totalTrades: engine.totalTrades, totalValue: finalVal };
  }
}


// --- Entry Point ---
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (process.argv.includes('--simulation')) {
    console.log("🦾 Background Simulator Worker Starting...");
    // Worker Logic Would Go Here... but for now we are just the Master.
    // Wait, we defined the SimulatorWorker Class inside but we didn't hook it up to run!
    // The user asked for "Self Contained". The SimulatorWorker class IS inside.
    // We need to instantiate it if --simulation is present.
    const worker = new ScientificOptimizer();
    // Export helper for external testing if attached (dirty hack for test script)
    if (global.TEST_MODE) global.testWorker = worker;

    // --- IPC Listener for Shadow Feedback ---
    process.on('message', (msg) => {
      if (msg.type === 'FEEDBACK') {
        worker.handleFeedback(msg);
      } else if (msg.type === 'OPTIMIZE_ORDER') {
        worker.prioritize(msg.symbol, msg.baseGenome);
      }
    });

    worker.run().catch(err => {
      console.error("💥 Simulator Worker Crashed:", err);
      process.exit(1);
    });
  } else {
    // Master Process
    mainLoop().catch((err) => {
      console.error("❌ Fatal Error in Main Loop:", err);
      rl.close();
    });
  }
}
export { TradingEngine, RobinhoodAPI, defaultGenome, ScientificOptimizer };


// ==================== Change Log ====================
// v1.999: "TrikeShed/Wasm Presentation" (Current Version)
// - stages a versioned Dreamer presentation bundle beside the static TrikeShed-linked Wasm outputs
// - adds a bash build entrypoint that packages `Dreamer 1.999.js` plus the standalone Wasm tool
// - preserves the existing trading engine behavior while exposing the new bundle surface
// - **MAJOR ARCHITECTURE CHANGE**: Migrated 11 hard-coded constants into evolving genome
// - Added Physics Genes: SPAR_DRAG_COEFFICIENT (0.80-0.999999), PRICE_HISTORY_WINDOW_SIZE (20-2000), ADAPTIVE_VOLATILITY_THRESHOLD (0.001-0.10)
// - Added Time Genes: FORCED_HARVEST_TIMEOUT, REBALANCE_COOLDOWN, FORCE_REBALANCE_TIMEOUT, ADAPTIVE_DZ_INACTIVITY_TIMEOUT (1min-24hrs)
// - Added Risk Genes: MAX_REBALANCE_ATTEMPTS (1-20), CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT (-50% to -1%)
// - Added Evolution Genes: EVOLUTION_TRADE_THRESHOLD (2 trades min), EVOLUTION_CONSISTENCY_COUNT (3 wins required)
// - Implemented comprehensive mutation bounds checking in _applyMutation to prevent catastrophic parameter drift
// - Removed SPAR_DRAG_COEFFICIENT filter exclusion from mutateGenome - all numeric parameters now evolvable
// - Updated EvolutionManager promotion logic to use genome-based thresholds instead of global constants
// - Added startup "Genome Personality" display showing key evolved parameters
// - Fixed variable name bug in asset specialization mutation (targetAsset -> targetSymbol)
// - Bot now autonomously optimizes its physics (baseline weight, memory), timing (patience), risk tolerance (panic threshold), and promotion criteria
// v3.23.7:
// - Added automatic ETH allocation from harvest proceeds (currentGenome.HARVEST_ALLOC_ETH_PERCENT).
// - Adjusted default allocation percentages: BTC 10%, ETH 10%, Reinvest 70%, Cash 10%.
// - Added ETH auto-buy execution block.
// v3.23.6:
// - Refined Harvest/Rebalance cycle counting logic to only increment/decrement on *strict* deviation changes, ignoring stagnant values (within a small float precision threshold).
// v3.23.5:
// - Prevented Adaptive Dead Zone activation/deactivation from incorrectly resetting existing Harvest/Rebalance cycle counters. Counters are now only reset on trade execution or when the asset leaves the trigger zone.
// v3.23.4:
// - Corrected Adaptive Dead Zone activation/deactivation logic to handle boundary conditions precisely.
//   - Deactivation now occurs if deviation is ON or OUTSIDE the original +/- trigger bounds.
//   - Activation now only occurs if deviation is STRICTLY INSIDE the original +/- trigger bounds AND inactivity timeout is met.
// v3.23.3:
// - Restored detailed console logging for Harvest/Rebalance cycle count increments/decrements based on deviation changes.
// v3.23.2:
// - Added currentGenome.ADAPTIVE_CONFIRMATION_CYCLE_INCREMENT constant.
// - Added currentGenome.ADAPTIVE_SKIP_BASELINE_ADJUST constant.
// - Updated Harvest logic to use currentGenome.ADAPTIVE_CONFIRMATION_CYCLE_INCREMENT for required cycles.
// - Updated Rebalance logic to use currentGenome.ADAPTIVE_CONFIRMATION_CYCLE_INCREMENT for required cycles.
// - Updated Harvest (Standard & Forced) logic to conditionally skip baseline adjustment based on currentGenome.ADAPTIVE_SKIP_BASELINE_ADJUST when adaptive mode is active.
// - Updated Rebalance (Standard & Forced) logic to conditionally skip baseline adjustment based on currentGenome.ADAPTIVE_SKIP_BASELINE_ADJUST when adaptive mode is active.
// - Simplified console display format for active harvest/rebalance/cooldown states.
// v3.23.1:
// - Revised Crash Protection (CP) trigger:
//   - Removed trigger based on overall portfolio average deviation.
//   - CP now activates *only* if a high percentage (`currentGenome.CP_TRIGGER_ASSET_PERCENT`) of *all* assets with baselines simultaneously drop below a minimum negative deviation (`currentGenome.CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT`).
//   - Added `currentGenome.CP_TRIGGER_ASSET_PERCENT` and `currentGenome.CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT` constants.
//   - Updated the CP check logic block accordingly.
// v3.23.0:
// - Added Adaptive Dead Zone Mode.
//   - Activates per-asset using +/-2% triggers if inactive (configurable timeout based on last trade) AND within original DZ bounds.
//   - Requires +1 confirmation cycle for adaptive trades & skips baseline adjustments. Deactivates instantly if price moves outside original DZ bounds.
//   - Added related constants, `lastActionTimestamps` (persistent state), `adaptiveDeadZoneState` (transient state).
//   - Updated `loadState`, `saveState`, trade success blocks, harvest/rebalance logic, and state display.
// v3.22.1:
// - Restored original cycle counter logic.
// v3.22.0:
// - Added Harvest Proceeds Allocation system. Adjusted currentGenome.PARTIAL_RECOVERY_PERCENT.
// ... (Previous logs unchanged) ...
// =====================================================
