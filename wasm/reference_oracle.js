/**
 * Deterministic Reference Specification (Oracle) for Dreamer WASM Kernel
 * Version: 1.0.0
 * Precision: binary64 (IEEE 754 double-precision)
 * 
 * This module provides reference implementations for all promoted calculations
 * in the WASM kernel. It serves as the oracle for kernel verification.
 */

'use strict';

// ============================================================================
// IEEE 754 Binary64 Constants and Utilities
// ============================================================================

const IEEE754 = Object.freeze({
    POSITIVE_ZERO: 0.0,
    NEGATIVE_ZERO: -0.0,
    POSITIVE_INFINITY: Infinity,
    NEGATIVE_INFINITY: -Infinity,
    
    // Smallest positive normal number: 2^-1022
    MIN_POSITIVE_NORMAL: Math.pow(2, -1022),
    
    // Smallest positive subnormal number: 2^-1074
    MIN_POSITIVE_SUBNORMAL: Math.pow(2, -1074),
    
    // Largest finite number: (2 - 2^-52) * 2^1023
    MAX_FINITE: (2.0 - Math.pow(2, -52)) * Math.pow(2, 1023),
    
    // Machine epsilon: 2^-52
    EPSILON: Math.pow(2, -52),
    
    /**
     * Check if value is -0.0 (negative zero)
     */
    isSignedZero(x) {
        return x === 0.0 && Object.is(x, -0.0);
    },
    
    /**
     * Check if value is +0.0 (positive zero)
     */
    isPositiveZero(x) {
        return x === 0.0 && Object.is(x, 0.0);
    },
    
    /**
     * Check if value is subnormal (denormal)
     */
    isSubnormal(x) {
        if (!Number.isFinite(x) || x === 0.0) {
            return false;
        }
        const absX = Math.abs(x);
        return absX > 0.0 && absX < IEEE754.MIN_POSITIVE_NORMAL;
    },
    
    /**
     * Get hex representation of float64
     */
    floatToHex(f) {
        const buf = new ArrayBuffer(8);
        const view = new DataView(buf);
        view.setFloat64(0, f);
        return view.getBigUint64(0).toString(16).padStart(16, '0');
    },
    
    /**
     * Check if two floats have identical bit representations
     */
    sameBits(a, b) {
        if (Number.isNaN(a) && Number.isNaN(b)) {
            // JavaScript doesn't distinguish different NaN bit patterns
            return true;
        }
        if (Object.is(a, b)) {
            return true;
        }
        // Check if both -0.0
        if (Object.is(a, -0.0) && Object.is(b, -0.0)) {
            return true;
        }
        return false;
    }
});

// ============================================================================
// Regime Encoding (must match dreamer_kernel.rs)
// ============================================================================

const Regime = Object.freeze({
    UNKNOWN: 0.0,
    CRAB_CHOP: 1.0,
    BULL_RUSH: 2.0,
    BEAR_CRASH: 3.0,
    STEADY_GROWTH: 4.0,
    VOLATILE_CHOP: 5.0,
});

// ============================================================================
// Reference Implementation: Rolling Window Features
// ============================================================================

/**
 * Compute rolling window features (reference implementation)
 * 
 * @param {Float64Array} prices - Price array
 * @param {Float64Array} volumes - Volume array
 * @param {number} windowSize - Window size
 * @returns {Object} WindowFeatures result
 */
function computeWindowFeaturesReference(prices, volumes, windowSize) {
    const n = Math.min(prices.length, volumes.length, windowSize);
    
    if (n === 0) {
        const nan = NaN;
        return {
            meanPrice: nan,
            priceVariance: nan,
            vwap: nan,
            latestPrice: nan,
            priceMomentum: nan,
            meanVolume: nan
        };
    }
    
    // Use last n elements
    const startIdx = prices.length - n;
    const windowPrices = prices.slice(startIdx);
    const windowVolumes = volumes.slice(startIdx);
    
    // Accumulate statistics
    const nf = n;
    let sumPrice = 0.0;
    let sumVolume = 0.0;
    let notional = 0.0;
    
    for (let i = 0; i < n; i++) {
        const price = windowPrices[i];
        const volume = windowVolumes[i];
        sumPrice += price;
        sumVolume += volume;
        notional += price * volume;
    }
    
    // Compute variance using Welford's algorithm for numerical stability
    let mean = 0.0;
    let m2 = 0.0;
    for (let i = 0; i < n; i++) {
        const price = windowPrices[i];
        const delta = price - mean;
        mean += delta / (i + 1);
        const delta2 = price - mean;
        m2 += delta * delta2;
    }
    
    const variance = n > 1 ? m2 / nf : 0.0;
    
    // Compute other features
    const meanPrice = sumPrice / nf;
    const vwap = sumVolume !== 0.0 ? notional / sumVolume : NaN;
    const oldestPrice = windowPrices[0];
    const latestPrice = windowPrices[n - 1];
    const priceMomentum = latestPrice - oldestPrice;
    const meanVolume = sumVolume / nf;
    
    return {
        meanPrice: meanPrice,
        priceVariance: variance,
        vwap: vwap,
        latestPrice: latestPrice,
        priceMomentum: priceMomentum,
        meanVolume: meanVolume
    };
}

// ============================================================================
// Reference Implementation: Portfolio Features
// ============================================================================

/**
 * Compute portfolio features (reference implementation)
 * 
 * @param {Float64Array} values - Asset values
 * @param {Float64Array} baselines - Asset baselines
 * @param {number} cashBalance - Cash balance
 * @param {number} harvestTrigger - Harvest trigger threshold
 * @param {number} rebalanceTrigger - Rebalance trigger threshold
 * @param {number} cpTriggerAssetPercent - Crash protection asset percentage
 * @param {number} cpTriggerMinNegativeDev - Crash protection min negative deviation
 * @returns {Object} PortfolioFeatures result
 */
function computePortfolioFeaturesReference(
    values,
    baselines,
    cashBalance,
    harvestTrigger,
    rebalanceTrigger,
    cpTriggerAssetPercent,
    cpTriggerMinNegativeDev
) {
    const assetCount = values.length;
    
    if (baselines.length !== assetCount) {
        throw new Error("values and baselines must have same length");
    }
    
    const deviations = new Float64Array(assetCount);
    const harvestCandidates = new Float64Array(assetCount);
    const rebalanceCandidates = new Float64Array(assetCount);
    
    let totalBaselineDiff = 0.0;
    let totalManagedBaseline = 0.0;
    let decliningCount = 0.0;
    
    for (let i = 0; i < assetCount; i++) {
        const value = values[i];
        const baseline = baselines[i];
        
        // Compute deviation
        let deviation;
        if (baseline > 0.0) {
            deviation = (value - baseline) / baseline;
        } else {
            deviation = 0.0;
        }
        deviations[i] = deviation;
        
        // Harvest candidate
        harvestCandidates[i] = deviation >= harvestTrigger ? 1.0 : 0.0;
        
        // Rebalance candidate
        rebalanceCandidates[i] = deviation <= -rebalanceTrigger ? 1.0 : 0.0;
        
        // Crash protection counting
        if (baseline > 0.0) {
            totalBaselineDiff += (value - baseline);
            totalManagedBaseline += baseline;
            
            if (deviation <= cpTriggerMinNegativeDev) {
                decliningCount += 1.0;
            }
        }
    }
    
    // Compute aggregate metrics
    const deviationPercent = totalManagedBaseline > 0.0 
        ? (totalBaselineDiff / totalManagedBaseline) * 100.0 
        : 0.0;
    
    const n = assetCount;
    let crashActive;
    if (totalManagedBaseline > 0.0) {
        const percentDeclining = decliningCount / n;
        crashActive = percentDeclining >= cpTriggerAssetPercent ? 1.0 : 0.0;
    } else {
        crashActive = 0.0;
    }
    
    return {
        deviationPercent: deviationPercent,
        crashActive: crashActive,
        decliningCount: decliningCount,
        managedBaseline: totalManagedBaseline,
        baselineDiff: totalBaselineDiff,
        deviations: deviations,
        harvestCandidates: harvestCandidates,
        rebalanceCandidates: rebalanceCandidates
    };
}

// ============================================================================
// Reference Implementation: Defect Scan
// ============================================================================

/**
 * Scan for defects (reference implementation)
 * 
 * @param {Float64Array} prices - Price history
 * @param {number} rebalanceTrigger - Rebalance trigger (negative)
 * @param {number} crashThreshold - Crash threshold
 * @returns {Object} DefectScanResult
 */
function scanForDefectsReference(prices, rebalanceTrigger, crashThreshold) {
    const priceCount = prices.length;
    
    if (priceCount < 2) {
        return {
            isDefective: 0.0,
            maxDrawdown: 0.0,
            triggerHits: 0.0
        };
    }
    
    let maxPrice = prices[0];
    let isDefective = 0.0;
    let maxDrawdown = 0.0;
    let triggerHits = 0.0;
    
    for (let i = 0; i < priceCount; i++) {
        const currentPrice = prices[i];
        
        if (currentPrice > maxPrice) {
            maxPrice = currentPrice;
        }
        
        // Check if trigger condition is met
        let deviation;
        if (maxPrice !== 0.0) {
            deviation = (currentPrice - maxPrice) / maxPrice;
        } else {
            deviation = NaN;
        }
        
        if (deviation < rebalanceTrigger) {
            triggerHits += 1.0;
            
            // Look ahead for crash
            let minFuturePrice = currentPrice;
            for (let j = i + 1; j < priceCount; j++) {
                if (prices[j] < minFuturePrice) {
                    minFuturePrice = prices[j];
                }
            }
            
            let subsequentDrop;
            if (currentPrice !== 0.0) {
                subsequentDrop = (minFuturePrice - currentPrice) / currentPrice;
            } else {
                subsequentDrop = NaN;
            }
            
            if (subsequentDrop < -crashThreshold) {
                isDefective = 1.0;
                const drawdown = -subsequentDrop;
                if (drawdown > maxDrawdown) {
                    maxDrawdown = drawdown;
                }
            }
        }
    }
    
    return {
        isDefective: isDefective,
        maxDrawdown: maxDrawdown,
        triggerHits: triggerHits
    };
}

// ============================================================================
// Reference Implementation: Regime Detection
// ============================================================================

/**
 * Compute market regime (reference implementation)
 * 
 * @param {Float64Array} history - Price history
 * @param {number} currentPrice - Current price
 * @param {number} startPrice - Start price
 * @returns {Object} RegimeResult
 */
function computeRegimeReference(history, currentPrice, startPrice) {
    const historyLen = history.length;
    
    if (historyLen < 50 || startPrice <= 0.0) {
        return {
            regime: Regime.UNKNOWN,
            roi: 0.0,
            volatility: 0.0,
            mean: 0.0
        };
    }
    
    const roi = (currentPrice - startPrice) / startPrice;
    
    // Compute mean
    let total = 0.0;
    for (let i = 0; i < historyLen; i++) {
        total += history[i];
    }
    const mean = total / historyLen;
    
    // Compute variance
    let sumSqDiff = 0.0;
    for (let i = 0; i < historyLen; i++) {
        const diff = history[i] - mean;
        sumSqDiff += diff * diff;
    }
    const variance = sumSqDiff / historyLen;
    const volatility = mean > 0.0 ? Math.sqrt(variance) / mean : 0.0;
    
    // Classify regime
    let regime;
    if (roi > 0.05 && volatility > 0.02) {
        regime = Regime.BULL_RUSH;
    } else if (roi < -0.05 && volatility > 0.02) {
        regime = Regime.BEAR_CRASH;
    } else if (roi > 0.02 && volatility < 0.01) {
        regime = Regime.STEADY_GROWTH;
    } else if (volatility > 0.05) {
        regime = Regime.VOLATILE_CHOP;
    } else {
        regime = Regime.CRAB_CHOP;
    }
    
    return {
        regime: regime,
        roi: roi,
        volatility: volatility,
        mean: mean
    };
}

// ============================================================================
// Test Corpus: Normal Market Paths
// ============================================================================

const NORMAL_MARKET_PATHS = Object.freeze({
    simpleUptrend: {
        description: "Simple uptrend with minimal volatility",
        prices: new Float64Array(Array.from({length: 100}, (_, i) => 100.0 + i * 0.5)),
        volumes: new Float64Array(Array.from({length: 100}, (_, i) => 1000.0 + i * 10)),
        windowSize: 20
    },
    simpleDowntrend: {
        description: "Simple downtrend with minimal volatility",
        prices: new Float64Array(Array.from({length: 100}, (_, i) => 150.0 - i * 0.3)),
        volumes: new Float64Array(Array.from({length: 100}, (_, i) => 5000.0 - i * 20)),
        windowSize: 20
    },
    sidewaysMarket: {
        description: "Sideways market with oscillation",
        prices: new Float64Array(Array.from({length: 100}, (_, i) => 100.0 + 5.0 * Math.sin(i * 0.2))),
        volumes: new Float64Array(100).fill(2000.0),
        windowSize: 20
    },
    volatileMarket: {
        description: "High volatility market",
        prices: new Float64Array(Array.from({length: 100}, (_, i) => 100.0 + 15.0 * Math.sin(i * 0.5) + 5.0 * (i % 3 - 1))),
        volumes: new Float64Array(Array.from({length: 100}, (_, i) => 1000.0 + 500.0 * (i % 5))),
        windowSize: 20
    }
});

// ============================================================================
// Test Corpus: 2-Degree-of-Contact Boundary Scenarios
// ============================================================================

const CONTACT_2_BOUNDARIES = Object.freeze({
    windowExactlyFull: {
        description: "Window size equals data length",
        prices: new Float64Array(Array.from({length: 20}, (_, i) => 100.0 + i * 1.0)),
        volumes: new Float64Array(20).fill(1000.0),
        windowSize: 20
    },
    windowOneUnder: {
        description: "Window size one less than data length",
        prices: new Float64Array(Array.from({length: 21}, (_, i) => 100.0 + i * 1.0)),
        volumes: new Float64Array(21).fill(1000.0),
        windowSize: 20
    },
    windowLargerThanData: {
        description: "Window size larger than available data",
        prices: new Float64Array(Array.from({length: 10}, (_, i) => 100.0 + i * 1.0)),
        volumes: new Float64Array(10).fill(1000.0),
        windowSize: 20
    },
    emptyWindow: {
        description: "Empty data",
        prices: new Float64Array(0),
        volumes: new Float64Array(0),
        windowSize: 10
    },
    singlePoint: {
        description: "Single data point",
        prices: new Float64Array([100.0]),
        volumes: new Float64Array([1000.0]),
        windowSize: 10
    }
});

// ============================================================================
// Test Corpus: IEEE 754 Corner Cases
// ============================================================================

const IEEE_754_CORNER_CASES = Object.freeze({
    positiveInfinityPrice: {
        description: "Positive infinity in price data",
        prices: new Float64Array([100.0, Infinity, 102.0]),
        volumes: new Float64Array([1000.0, 1000.0, 1000.0]),
        windowSize: 3
    },
    negativeInfinityPrice: {
        description: "Negative infinity in price data",
        prices: new Float64Array([100.0, -Infinity, 102.0]),
        volumes: new Float64Array([1000.0, 1000.0, 1000.0]),
        windowSize: 3
    },
    nanInPrices: {
        description: "NaN in price data",
        prices: new Float64Array([100.0, NaN, 102.0]),
        volumes: new Float64Array([1000.0, 1000.0, 1000.0]),
        windowSize: 3
    },
    nanInVolumes: {
        description: "NaN in volume data",
        prices: new Float64Array([100.0, 101.0, 102.0]),
        volumes: new Float64Array([1000.0, NaN, 1000.0]),
        windowSize: 3
    },
    zeroVolume: {
        description: "Zero volume (VWAP boundary)",
        prices: new Float64Array([100.0, 101.0, 102.0]),
        volumes: new Float64Array([0.0, 0.0, 0.0]),
        windowSize: 3
    },
    negativeZero: {
        description: "Negative zero values",
        prices: new Float64Array([100.0, -0.0, 102.0]),
        volumes: new Float64Array([1000.0, 1000.0, 1000.0]),
        windowSize: 3
    },
    subnormalPrices: {
        description: "Subnormal (denormal) prices",
        prices: new Float64Array([
            IEEE754.MIN_POSITIVE_SUBNORMAL,
            IEEE754.MIN_POSITIVE_SUBNORMAL * 2,
            IEEE754.MIN_POSITIVE_SUBNORMAL * 3
        ]),
        volumes: new Float64Array([1000.0, 1000.0, 1000.0]),
        windowSize: 3
    }
});

// ============================================================================
// Reference Oracle Class
// ============================================================================

class ReferenceOracle {
    constructor() {
        this.expectedResults = this._generateExpectedResults();
    }
    
    _generateExpectedResults() {
        const results = {
            windowFeatures: {}
        };
        
        // Generate expected results for normal paths
        for (const [name, data] of Object.entries(NORMAL_MARKET_PATHS)) {
            if (data.prices && data.volumes && data.windowSize) {
                results.windowFeatures[name] = computeWindowFeaturesReference(
                    data.prices,
                    data.volumes,
                    data.windowSize
                );
            }
        }
        
        // Generate expected results for boundary cases
        for (const [name, data] of Object.entries(CONTACT_2_BOUNDARIES)) {
            if (data.prices && data.volumes && data.windowSize) {
                results.windowFeatures[name] = computeWindowFeaturesReference(
                    data.prices,
                    data.volumes,
                    data.windowSize
                );
            }
        }
        
        // Generate expected results for IEEE 754 corner cases
        for (const [name, data] of Object.entries(IEEE_754_CORNER_CASES)) {
            if (data.prices && data.volumes && data.windowSize) {
                results.windowFeatures[name] = computeWindowFeaturesReference(
                    data.prices,
                    data.volumes,
                    data.windowSize
                );
            }
        }
        
        return results;
    }
    
    getExpectedWindowFeatures(testCase) {
        return this.expectedResults.windowFeatures[testCase];
    }
    
    /**
     * Verify actual window features against expected
     * 
     * @param {string} testCase - Test case name
     * @param {Object} actual - Actual result
     * @param {number} tolerance - Tolerance for comparison
     * @returns {Object} {pass: boolean, failures: string[]}
     */
    verifyWindowFeatures(testCase, actual, tolerance = 1e-15) {
        const expected = this.getExpectedWindowFeatures(testCase);
        if (!expected) {
            return { pass: false, failures: [`No expected results for test case: ${testCase}`] };
        }
        
        const fields = ['meanPrice', 'priceVariance', 'vwap', 'latestPrice', 'priceMomentum', 'meanVolume'];
        const failures = [];
        
        for (const field of fields) {
            const expVal = expected[field];
            const actVal = actual[field];
            
            // Handle NaN comparison
            if (Number.isNaN(expVal) && Number.isNaN(actVal)) {
                continue;
            }
            
            // Handle infinite values
            if (!Number.isFinite(expVal) && !Number.isFinite(actVal)) {
                if ((expVal > 0) === (actVal > 0)) {  // Same sign
                    continue;
                }
            }
            
            // Finite comparison
            const relDiff = Math.abs(expVal - actVal) / (Math.abs(expVal) + Math.abs(actVal) + 1e-300);
            if (relDiff > tolerance && Math.abs(expVal - actVal) > tolerance) {
                failures.push(`${field}: expected ${expVal}, got ${actVal}`);
            }
        }
        
        return { pass: failures.length === 0, failures };
    }
    
    computeReference(kernelType, inputs) {
        switch (kernelType) {
            case 'windowFeatures':
                return computeWindowFeaturesReference(
                    inputs.prices,
                    inputs.volumes,
                    inputs.windowSize || 20
                );
            case 'portfolioFeatures':
                return computePortfolioFeaturesReference(
                    inputs.values,
                    inputs.baselines,
                    inputs.cashBalance || 0.0,
                    inputs.harvestTrigger || 0.0,
                    inputs.rebalanceTrigger || 0.0,
                    inputs.cpTriggerAssetPercent || 0.5,
                    inputs.cpTriggerMinNegativeDev || -0.1
                );
            case 'defectScan':
                return scanForDefectsReference(
                    inputs.prices,
                    inputs.rebalanceTrigger,
                    inputs.crashThreshold
                );
            case 'regime':
                return computeRegimeReference(
                    inputs.history,
                    inputs.currentPrice,
                    inputs.startPrice
                );
            default:
                throw new Error(`Unknown kernel type: ${kernelType}`);
        }
    }
}

// ============================================================================
// Serialization for Cross-Language Compatibility
// ============================================================================

/**
 * Serialize a reference result to a JSON-compatible object
 */
function serializeReferenceResult(result) {
    function serializeValue(v) {
        if (typeof v === 'number') {
            return {
                value: v,
                hex: IEEE754.floatToHex(v),
                isNan: Number.isNaN(v),
                isInf: !Number.isFinite(v) && !Number.isNaN(v),
                isNegZero: IEEE754.isSignedZero(v)
            };
        } else if (v instanceof Float64Array) {
            return Array.from(v).map(serializeValue);
        } else if (typeof v === 'object' && v !== null) {
            const obj = {};
            for (const [key, val] of Object.entries(v)) {
                obj[key] = serializeValue(val);
            }
            return obj;
        }
        return v;
    }
    
    return serializeValue(result);
}

// ============================================================================
// Exports
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    // Node.js exports
    module.exports = {
        IEEE754,
        Regime,
        computeWindowFeaturesReference,
        computePortfolioFeaturesReference,
        scanForDefectsReference,
        computeRegimeReference,
        ReferenceOracle,
        NORMAL_MARKET_PATHS,
        CONTACT_2_BOUNDARIES,
        IEEE_754_CORNER_CASES,
        serializeReferenceResult
    };
}

// Browser/global exports
if (typeof window !== 'undefined') {
    window.ReferenceOracle = ReferenceOracle;
    window.IEEE754 = IEEE754;
    window.Regime = Regime;
}
