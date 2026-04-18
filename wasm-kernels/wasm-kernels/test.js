// test.js

const fs = require('fs');
const path = require('path');

let FinancialKernels;

try {
  FinancialKernels = require('./financial_kernels.js');
} catch (error) {
  console.error('Error loading WASM module:', error);
  process.exit(1);
}

function generateTestData(length) {
  const prices = new Float64Array(length);
  const high = new Float64Array(length);
  const low = new Float64Array(length);
  const close = new Float64Array(length);
  
  for (let i = 0; i < length; i++) {
    prices[i] = 100.0 + 20.0 * Math.sin(i * 0.1) + 0.5 * Math.random();
    high[i] = prices[i] + 1.0 + 2.0 * Math.random();
    low[i] = prices[i] - 1.0 - 2.0 * Math.random();
    close[i] = prices[i] + 1.0 * Math.random() - 0.5;
  }
  
  return { prices, high, low, close };
}

function arraysEqual(a, b, tolerance = 0) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

function benchmark(name, fn, iterations = 1000) {
  const start = process.hrtime.bigint();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = process.hrtime.bigint();
  const totalTime = Number(end - start) / 1e6;
  const avgTime = totalTime / iterations;
  console.log(`${name}: ${avgTime.toFixed(3)}ms avg, ${totalTime.toFixed(3)}ms total (${iterations} iterations)`);
  return avgTime;
}

async function runTests() {
  console.log('Loading WASM module...');
  const Module = await FinancialKernels();
  
  const length = 10000;
  const period = 14;
  const { prices, high, low, close } = generateTestData(length);
  
  console.log(`\nTesting with ${length} data points, period ${period}`);
  console.log('First 5 prices:', Array.from(prices.slice(0, 5)).map(x => x.toFixed(6)).join(', '));
  
  console.log('\n=== SMA Test ===');
  const sma_output = new Float64Array(length);
  Module.ccall('sma_kernel', null, 
    ['number', 'number', 'number', 'number'],
    [prices.byteOffset, sma_output.byteOffset, length, period]
  );
  
  const sma_ref = new Float64Array(length);
  let sum = 0.0;
  for (let i = 0; i < period; i++) {
    sum += prices[i];
  }
  sma_ref[period - 1] = sum / period;
  
  for (let i = period; i < length; i++) {
    sum = sum - prices[i - period] + prices[i];
    sma_ref[i] = sum / period;
  }
  
  const smaEqual = arraysEqual(sma_output, sma_ref, 1e-15);
  console.log('SMA results match reference:', smaEqual);
  if (!smaEqual) {
    console.log('First 10 values - WASM:', Array.from(sma_output.slice(period-1, period+9)).map(x => x.toFixed(10)));
    console.log('First 10 values - Ref:  ', Array.from(sma_ref.slice(period-1, period+9)).map(x => x.toFixed(10)));
  }
  
  console.log('\n=== EMA Test ===');
  const ema_output = new Float64Array(length);
  Module.ccall('ema_kernel', null,
    ['number', 'number', 'number', 'number'],
    [prices.byteOffset, ema_output.byteOffset, length, period]
  );
  
  const ema_ref = new Float64Array(length);
  const alpha = 2.0 / (period + 1.0);
  ema_ref[0] = prices[0];
  for (let i = 1; i < length; i++) {
    ema_ref[i] = alpha * prices[i] + (1.0 - alpha) * ema_ref[i - 1];
  }
  
  const emaEqual = arraysEqual(ema_output, ema_ref, 1e-15);
  console.log('EMA results match reference:', emaEqual);
  
  console.log('\n=== RSI Test ===');
  const rsi_output = new Float64Array(length);
  Module.ccall('rsi_kernel', null,
    ['number', 'number', 'number', 'number'],
    [prices.byteOffset, rsi_output.byteOffset, length, period]
  );
  
  const rsi_ref = new Float64Array(length);
  let gain_sum = 0.0, loss_sum = 0.0;
  for (let i = 1; i <= period; i++) {
    const diff = prices[i] - prices[i - 1];
    if (diff > 0.0) gain_sum += diff;
    else loss_sum += -diff;
  }
  
  let avg_gain = gain_sum / period;
  let avg_loss = loss_sum / period;
  
  if (avg_loss === 0.0) {
    rsi_ref[period] = 100.0;
  } else {
    rsi_ref[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
  }
  
  for (let i = period + 1; i < length; i++) {
    const diff = prices[i] - prices[i - 1];
    const gain = diff > 0.0 ? diff : 0.0;
    const loss = diff < 0.0 ? -diff : 0.0;
    
    avg_gain = (avg_gain * (period - 1.0) + gain) / period;
    avg_loss = (avg_loss * (period - 1.0) + loss) / period;
    
    if (avg_loss === 0.0) {
      rsi_ref[i] = 100.0;
    } else {
      rsi_ref[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
    }
  }
  
  const rsiEqual = arraysEqual(rsi_output, rsi_ref, 1e-15);
  console.log('RSI results match reference:', rsiEqual);
  
  console.log('\n=== Performance Benchmarks ===');
  
  const benchmarkLength = 100000;
  const benchData = generateTestData(benchmarkLength);
  
  console.log('\nWarmup...');
  Module.ccall('sma_kernel', null,
    ['number', 'number', 'number', 'number'],
    [benchData.prices.byteOffset, sma_output.byteOffset, 1000, 14]
  );
  
  console.log('\nBenchmarks (100 iterations each):');
  
  const smaTime = benchmark('SMA', () => {
    Module.ccall('sma_kernel', null,
      ['number', 'number', 'number', 'number'],
      [benchData.prices.byteOffset, sma_output.byteOffset, benchmarkLength, 14]
    );
  });
  
  const emaTime = benchmark('EMA', () => {
    Module.ccall('ema_kernel', null,
      ['number', 'number', 'number', 'number'],
      [benchData.prices.byteOffset, ema_output.byteOffset, benchmarkLength, 14]
    );
  });
  
  const rsiTime = benchmark('RSI', () => {
    Module.ccall('rsi_kernel', null,
      ['number', 'number', 'number', 'number'],
      [benchData.prices.byteOffset, rsi_output.byteOffset, benchmarkLength, 14]
    );
  });
  
  console.log(`\nTotal benchmark data size: ${benchmarkLength} elements`);
  console.log(`SMA throughput: ${(benchmarkLength / smaTime * 1000 / 1e6).toFixed(2)} million elements/second`);
  console.log(`EMA throughput: ${(benchmarkLength / emaTime * 1000 / 1e6).toFixed(2)} million elements/second`);
  console.log(`RSI throughput: ${(benchmarkLength / rsiTime * 1000 / 1e6).toFixed(2)} million elements/second`);
  
  const allTestsPassed = smaEqual && emaEqual && rsiEqual;
  console.log('\n=== Test Summary ===');
  console.log(`All tests passed: ${allTestsPassed}`);
  
  return allTestsPassed ? 0 : 1;
}

if (require.main === module) {
  runTests().then(exitCode => {
    process.exit(exitCode);
  }).catch(error => {
    console.error('Test failed:', error);
    process.exit(1);
  });
}