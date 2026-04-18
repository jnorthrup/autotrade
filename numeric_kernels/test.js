// Test suite for WASM numeric kernels
// Verifies correctness and bit-identical results

const fs = require('fs');
const path = require('path');

// Reference implementations for comparison
class ReferenceIndicators {
    
    static sma(prices, period) {
        const result = new Array(prices.length).fill(0.0);
        if (period <= 0 || period > prices.length) return result;
        
        let sum = 0.0;
        for (let i = 0; i < period; i++) {
            sum += prices[i];
        }
        result[period - 1] = sum / period;
        
        for (let i = period; i < prices.length; i++) {
            sum = sum - prices[i - period] + prices[i];
            result[i] = sum / period;
        }
        
        return result;
    }
    
    static ema(prices, period) {
        const result = new Array(prices.length).fill(0.0);
        if (period <= 0 || period > prices.length) return result;
        
        const alpha = 2.0 / (period + 1.0);
        
        // Initialize with SMA
        let sum = 0.0;
        for (let i = 0; i < period; i++) {
            sum += prices[i];
        }
        result[period - 1] = sum / period;
        
        for (let i = period; i < prices.length; i++) {
            result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1];
        }
        
        return result;
    }
    
    static atr(highs, lows, closes, period) {
        const result = new Array(highs.length).fill(0.0);
        if (period <= 0 || period > highs.length) return result;
        
        const trValues = new Array(highs.length);
        trValues[0] = highs[0] - lows[0];
        
        for (let i = 1; i < highs.length; i++) {
            const hl = highs[i] - lows[i];
            const hc = Math.abs(highs[i] - closes[i - 1]);
            const lc = Math.abs(lows[i] - closes[i - 1]);
            trValues[i] = Math.max(hl, Math.max(hc, lc));
        }
        
        // ATR calculation using Wilder's smoothing
        const alpha = 1.0 / period;
        
        let sum = 0.0;
        for (let i = 0; i < period; i++) {
            sum += trValues[i];
        }
        result[period - 1] = sum / period;
        
        for (let i = period; i < highs.length; i++) {
            result[i] = alpha * trValues[i] + (1 - alpha) * result[i - 1];
        }
        
        return result;
    }
    
    static rsi(prices, period) {
        const result = new Array(prices.length).fill(0.0);
        if (period <= 0 || period >= prices.length) return result;
        
        const gains = new Array(prices.length).fill(0.0);
        const losses = new Array(prices.length).fill(0.0);
        
        const firstChange = prices[1] - prices[0];
        gains[1] = firstChange > 0 ? firstChange : 0.0;
        losses[1] = firstChange < 0 ? -firstChange : 0.0;
        
        for (let i = 2; i < prices.length; i++) {
            const change = prices[i] - prices[i - 1];
            gains[i] = change > 0 ? change : 0.0;
            losses[i] = change < 0 ? -change : 0.0;
        }
        
        // Calculate initial averages
        let avgGain = 0.0;
        let avgLoss = 0.0;
        for (let i = 1; i <= period; i++) {
            avgGain += gains[i];
            avgLoss += losses[i];
        }
        avgGain /= period;
        avgLoss /= period;
        
        if (avgLoss === 0.0) {
            result[period] = 100.0;
        } else {
            const rs = avgGain / avgLoss;
            result[period] = 100.0 - (100.0 / (1.0 + rs));
        }
        
        for (let i = period + 1; i < prices.length; i++) {
            avgGain = (avgGain * (period - 1) + gains[i]) / period;
            avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
            
            if (avgLoss === 0.0) {
                result[i] = 100.0;
            } else {
                const rs = avgGain / avgLoss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
        
        return result;
    }
    
    static macd(prices, fastPeriod, slowPeriod, signalPeriod) {
        const length = prices.length;
        const macdLine = new Array(length).fill(0.0);
        const signalLine = new Array(length).fill(0.0);
        const histogram = new Array(length).fill(0.0);
        
        if (fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0 || fastPeriod >= slowPeriod) {
            return { macdLine, signalLine, histogram };
        }
        
        const fastEMA = this.ema(prices, fastPeriod);
        const slowEMA = this.ema(prices, slowPeriod);
        
        const startIdx = Math.max(fastPeriod, slowPeriod) - 1;
        for (let i = 0; i < length; i++) {
            if (i < startIdx) {
                macdLine[i] = 0.0;
            } else {
                macdLine[i] = fastEMA[i] - slowEMA[i];
            }
        }
        
        const signalEMA = this.ema(macdLine, signalPeriod);
        
        const histogramStart = startIdx + signalPeriod - 1;
        for (let i = 0; i < length; i++) {
            signalLine[i] = signalEMA[i];
            if (i >= histogramStart) {
                histogram[i] = macdLine[i] - signalLine[i];
            }
        }
        
        return { macdLine, signalLine, histogram };
    }
    
    static fitness(prices, signals, riskFreeRate, windowSize) {
        const result = new Array(prices.length).fill(0.0);
        if (windowSize <= 0 || windowSize > prices.length) return result;
        
        const returns = new Array(prices.length);
        for (let i = 1; i < prices.length; i++) {
            returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
        }
        
        for (let i = windowSize; i < prices.length; i++) {
            let totalReturn = 0.0;
            let signalFollowingReturn = 0.0;
            let squaredReturns = 0.0;
            let signalTrades = 0;
            
            for (let j = i - windowSize + 1; j <= i; j++) {
                const ret = returns[j];
                totalReturn += ret;
                squaredReturns += ret * ret;
                
                if (signals[j] > 0.0) {
                    signalFollowingReturn += ret;
                    signalTrades++;
                }
            }
            
            const meanReturn = totalReturn / windowSize;
            const variance = (squaredReturns - (totalReturn * totalReturn) / windowSize) / (windowSize - 1);
            const volatility = Math.sqrt(Math.max(0.0, variance));
            
            let sharpeRatio = 0.0;
            if (volatility > 0.0) {
                sharpeRatio = (meanReturn - riskFreeRate / 252.0) / (volatility * Math.sqrt(252.0));
            }
            
            let signalEffectiveness = 0.0;
            if (signalTrades > 0) {
                signalEffectiveness = signalFollowingReturn / signalTrades;
            }
            
            result[i] = 0.6 * sharpeRatio + 0.4 * signalEffectiveness;
        }
        
        return result;
    }
}

// WASM memory management
class WASMInterface {
    constructor() {
        this.wasmModule = null;
        this.memory = null;
    }
    
    async initialize() {
        const wasmPath = path.join(__dirname, 'indicators.wasm');
        const wasmBuffer = fs.readFileSync(wasmPath);
        
        const importObject = {
            env: {
                memory: new WebAssembly.Memory({ initial: 256, maximum: 512 }),
                __memory_base: 0,
                __table_base: 0,
                emscripten_memcpy_big: (dest, src, num) => {
                    const memory = new Uint8Array(this.wasmModule.exports.memory.buffer);
                    memory.set(memory.subarray(src, src + num), dest);
                },
                emscripten_resize_heap: (delta) => {
                    return this.wasmModule.exports.memory.grow(Math.ceil(delta / 65536));
                }
            }
        };
        
        const module = await WebAssembly.compile(wasmBuffer);
        this.wasmModule = await WebAssembly.instantiate(module, importObject);
        this.memory = this.wasmModule.exports.memory;
        
        // Set up exported functions
        this.smaKernel = this.wasmModule.exports._sma_kernel;
        this.emaKernel = this.wasmModule.exports._ema_kernel;
        this.atrKernel = this.wasmModule.exports._atr_kernel;
        this.rsiKernel = this.wasmModule.exports._rsi_kernel;
        this.macdKernel = this.wasmModule.exports._macd_kernel;
        this.fitnessKernel = this.wasmModule.exports._fitness_kernel;
    }
    
    allocateFloat64Array(size) {
        const bytes = size * 8;
        const ptr = this.wasmModule.exports.malloc(bytes);
        return {
            ptr: ptr,
            size: size,
            getArray() {
                return new Float64Array(this.memory.buffer, this.ptr, this.size);
            }
        };
    }
    
    freeAllocation(allocation) {
        if (this.wasmModule.exports.free) {
            this.wasmModule.exports.free(allocation.ptr);
        }
    }
}

// Test suite
class TestSuite {
    constructor() {
        this.wasm = new WASMInterface();
        this.passed = 0;
        this.failed = 0;
        this.tolerance = 1e-15; // Double precision tolerance
    }
    
    async initialize() {
        await this.wasm.initialize();
    }
    
    assertEqual(actual, expected, message) {
        const diff = Math.abs(actual - expected);
        if (diff <= this.tolerance) {
            this.passed++;
            console.log(`✓ ${message}`);
        } else {
            this.failed++;
            console.log(`✗ ${message}: expected ${expected}, got ${actual}, diff: ${diff}`);
        }
    }
    
    assertArrayEqual(actual, expected, message) {
        let allEqual = true;
        let maxDiff = 0.0;
        
        for (let i = 0; i < actual.length; i++) {
            const diff = Math.abs(actual[i] - expected[i]);
            if (diff > this.tolerance) {
                allEqual = false;
                maxDiff = Math.max(maxDiff, diff);
            }
        }
        
        if (allEqual) {
            this.passed++;
            console.log(`✓ ${message}`);
        } else {
            this.failed++;
            console.log(`✗ ${message}: max difference ${maxDiff}`);
        }
    }
    
    testSMA() {
        console.log("\n=== Testing SMA ===");
        
        const prices = new Float64Array([10, 12, 15, 13, 16, 18, 20, 17, 19, 21]);
        const period = 3;
        
        const expected = ReferenceIndicators.sma(prices, period);
        
        // Allocate memory and call WASM
        const pricesAlloc = this.wasm.allocateFloat64Array(prices.length);
        const outputAlloc = this.wasm.allocateFloat64Array(prices.length);
        
        pricesAlloc.getArray().set(prices);
        
        this.wasm.smaKernel(pricesAlloc.ptr, outputAlloc.ptr, prices.length, period);
        
        const actual = new Float64Array(outputAlloc.getArray());
        
        this.assertArrayEqual(actual, expected, "SMA calculation");
        
        this.wasm.freeAllocation(pricesAlloc);
        this.wasm.freeAllocation(outputAlloc);
    }
    
    testEMA() {
        console.log("\n=== Testing EMA ===");
        
        const prices = new Float64Array([10, 12, 15, 13, 16, 18, 20, 17, 19, 21]);
        const period = 3;
        
        const expected = ReferenceIndicators.ema(prices, period);
        
        const pricesAlloc = this.wasm.allocateFloat64Array(prices.length);
        const outputAlloc = this.wasm.allocateFloat64Array(prices.length);
        
        pricesAlloc.getArray().set(prices);
        
        this.wasm.emaKernel(pricesAlloc.ptr, outputAlloc.ptr, prices.length, period);
        
        const actual = new Float64Array(outputAlloc.getArray());
        
        this.assertArrayEqual(actual, expected, "EMA calculation");
        
        this.wasm.freeAllocation(pricesAlloc);
        this.wasm.freeAllocation(outputAlloc);
    }
    
    testATR() {
        console.log("\n=== Testing ATR ===");
        
        const highs = new Float64Array([15, 16, 18, 17, 19, 20, 22, 21, 23, 24]);
        const lows = new Float64Array([10, 11, 13, 12, 14, 15, 17, 16, 18, 19]);
        const closes = new Float64Array([12, 14, 16, 15, 17, 18, 20, 19, 21, 22]);
        const period = 3;
        
        const expected = ReferenceIndicators.atr(highs, lows, closes, period);
        
        const highsAlloc = this.wasm.allocateFloat64Array(highs.length);
        const lowsAlloc = this.wasm.allocateFloat64Array(lows.length);
        const closesAlloc = this.wasm.allocateFloat64Array(closes.length);
        const outputAlloc = this.wasm.allocateFloat64Array(highs.length);
        
        highsAlloc.getArray().set(highs);
        lowsAlloc.getArray().set(lows);
        closesAlloc.getArray().set(closes);
        
        this.wasm.atrKernel(highsAlloc.ptr, lowsAlloc.ptr, closesAlloc.ptr, 
                           outputAlloc.ptr, highs.length, period);
        
        const actual = new Float64Array(outputAlloc.getArray());
        
        this.assertArrayEqual(actual, expected, "ATR calculation");
        
        this.wasm.freeAllocation(highsAlloc);
        this.wasm.freeAllocation(lowsAlloc);
        this.wasm.freeAllocation(closesAlloc);
        this.wasm.freeAllocation(outputAlloc);
    }
    
    testRSI() {
        console.log("\n=== Testing RSI ===");
        
        const prices = new Float64Array([10, 12, 15, 13, 16, 18, 20, 17, 19, 21]);
        const period = 5;
        
        const expected = ReferenceIndicators.rsi(prices, period);
        
        const pricesAlloc = this.wasm.allocateFloat64Array(prices.length);
        const outputAlloc = this.wasm.allocateFloat64Array(prices.length);
        
        pricesAlloc.getArray().set(prices);
        
        this.wasm.rsiKernel(pricesAlloc.ptr, outputAlloc.ptr, prices.length, period);
        
        const actual = new Float64Array(outputAlloc.getArray());
        
        this.assertArrayEqual(actual, expected, "RSI calculation");
        
        this.wasm.freeAllocation(pricesAlloc);
        this.wasm.freeAllocation(outputAlloc);
    }
    
    testMACD() {
        console.log("\n=== Testing MACD ===");
        
        const prices = new Float64Array([10, 12, 15, 13, 16, 18, 20, 17, 19, 21, 23, 25, 27, 29, 31]);
        const fastPeriod = 3;
        const slowPeriod = 6;
        const signalPeriod = 3;
        
        const expected = ReferenceIndicators.macd(prices, fastPeriod, slowPeriod, signalPeriod);
        
        const pricesAlloc = this.wasm.allocateFloat64Array(prices.length);
        const macdAlloc = this.wasm.allocateFloat64Array(prices.length);
        const signalAlloc = this.wasm.allocateFloat64Array(prices.length);
        const histAlloc = this.wasm.allocateFloat64Array(prices.length);
        
        pricesAlloc.getArray().set(prices);
        
        this.wasm.macdKernel(pricesAlloc.ptr, macdAlloc.ptr, signalAlloc.ptr, histAlloc.ptr, 
                           prices.length, fastPeriod, slowPeriod, signalPeriod);
        
        const actualMacd = new Float64Array(macdAlloc.getArray());
        const actualSignal = new Float64Array(signalAlloc.getArray());
        const actualHist = new Float64Array(histAlloc.getArray());
        
        this.assertArrayEqual(actualMacd, expected.macdLine, "MACD line");
        this.assertArrayEqual(actualSignal, expected.signalLine, "Signal line");
        this.assertArrayEqual(actualHist, expected.histogram, "Histogram");
        
        this.wasm.freeAllocation(pricesAlloc);
        this.wasm.freeAllocation(macdAlloc);
        this.wasm.freeAllocation(signalAlloc);
        this.wasm.freeAllocation(histAlloc);
    }
    
    testFitness() {
        console.log("\n=== Testing Fitness ===");
        
        const prices = new Float64Array([100, 102, 98, 105, 103, 107, 110, 108, 112, 115, 113, 117, 120, 118, 122]);
        const signals = new Float64Array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]);
        const riskFreeRate = 0.02;
        const windowSize = 5;
        
        const expected = ReferenceIndicators.fitness(prices, signals, riskFreeRate, windowSize);
        
        const pricesAlloc = this.wasm.allocateFloat64Array(prices.length);
        const signalsAlloc = this.wasm.allocateFloat64Array(signals.length);
        const fitnessAlloc = this.wasm.allocateFloat64Array(prices.length);
        
        pricesAlloc.getArray().set(prices);
        signalsAlloc.getArray().set(signals);
        
        this.wasm.fitnessKernel(pricesAlloc.ptr, signalsAlloc.ptr, fitnessAlloc.ptr, 
                              prices.length, riskFreeRate, windowSize);
        
        const actual = new Float64Array(fitnessAlloc.getArray());
        
        this.assertArrayEqual(actual, expected, "Fitness calculation");
        
        this.wasm.freeAllocation(pricesAlloc);
        this.wasm.freeAllocation(signalsAlloc);
        this.wasm.freeAllocation(fitnessAlloc);
    }
    
    printSummary() {
        console.log("\n=== Test Summary ===");
        console.log(`Total tests: ${this.passed + this.failed}`);
        console.log(`Passed: ${this.passed}`);
        console.log(`Failed: ${this.failed}`);
        console.log(`Success rate: ${(this.passed / (this.passed + this.failed) * 100).toFixed(1)}%`);
        
        if (this.failed === 0) {
            console.log("\n🎉 All tests passed! Bit-identical results confirmed.");
        } else {
            console.log("\n❌ Some tests failed. Check the differences above.");
        }
    }
}

// Main test runner
async function main() {
    console.log("=== WASM Numeric Kernels Test Suite ===");
    console.log("Testing bit-identical results with double precision...");
    
    const testSuite = new TestSuite();
    
    try {
        await testSuite.initialize();
        
        testSuite.testSMA();
        testSuite.testEMA();
        testSuite.testATR();
        testSuite.testRSI();
        testSuite.testMACD();
        testSuite.testFitness();
        
        testSuite.printSummary();
        
        process.exit(testSuite.failed > 0 ? 1 : 0);
        
    } catch (error) {
        console.error("Test error:", error);
        process.exit(1);
    }
}

if (require.main === module) {
    main().catch(console.error);
}