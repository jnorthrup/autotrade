'use strict';

const VALIDATION_PLAN = Object.freeze({
  title: 'Validation plan for numerical equivalence and throughput vs the current JS path',
  baselinePath: 'current JS Float64 path',
  scope: Object.freeze([
    'rolling-window feature extraction over contiguous and wrapped ring-buffer spans',
    'batch evaluation across series/window pairs',
    'indicator workloads built from the current JS path',
  ]),
  benchmarkCases: Object.freeze([
    Object.freeze({
      id: 'rolling-window-steady-state',
      workload: 'single-series rolling-window feature extraction',
      representativeOf: 'writeWindowFeatures on ring-buffer resident price and volume slabs',
      dataShape: '1 series, 16k ticks, 256-sample ring buffer, windows 5/20/64/128',
      comparesAgainst: 'current JS path',
      successMetric: 'median throughput in ops/sec and ns/op',
    }),
    Object.freeze({
      id: 'rolling-window-wraparound',
      workload: 'rolling-window reduction after ring-buffer wrap',
      representativeOf: 'tail-span and head-span reduction over a wrapped window',
      dataShape: '1 series, 16k ticks, 128-sample ring buffer, windows 20/64/96',
      comparesAgainst: 'current JS path',
      successMetric: 'median throughput plus exact wrap correctness',
    }),
    Object.freeze({
      id: 'batch-series-window-scan',
      workload: 'batched feature extraction across many series/window pairs',
      representativeOf: 'evaluateBatch with fixed FEATURE_COUNT stride output',
      dataShape: '256 series, 65k evaluated rows, windows 5/20/64/128',
      comparesAgainst: 'current JS batch path',
      successMetric: 'median throughput per evaluated row and allocation rate',
    }),
    Object.freeze({
      id: 'indicator-bundle',
      workload: 'indicator workload from the current JS path',
      representativeOf: 'SMA, EMA, ATR, RSI, MACD, VWAP, and momentum style rolling transforms',
      dataShape: '32 series, 50k bars, OHLCV-style price and volume traces, windows 10/12/14/20/26/60',
      comparesAgainst: 'current JS indicator code path',
      successMetric: 'median throughput and numerical error versus baseline',
    }),
  ]),
  numericalEquivalence: Object.freeze({
    bitForBitExpectations: Object.freeze([
      'Float64 ingress and egress through the adapter and arena views must be bit-for-bit identical to the source values',
      'Direct copy outputs, including latest_price, must match bit-for-bit',
      'NaN and Infinity classifications must match the JS baseline exactly',
    ]),
    tolerances: Object.freeze({
      absolute: 1e-12,
      relative: 1e-12,
      ulp: 1,
    }),
    comparisonRules: Object.freeze([
      'Use bit-for-bit comparison for direct copies and boundary round-trips',
      'Use absolute and relative tolerance for reductions such as mean, variance, VWAP, momentum, and indicator aggregates',
      'Treat a candidate NaN as acceptable only when the baseline is also NaN in the same output slot',
      'Reject any sign flip, category mismatch, or tolerance miss as a precision regression',
    ]),
  }),
  autoVectorizationEvidence: Object.freeze({
    requiredArtifacts: Object.freeze([
      'WASM build compiled with -C target-feature=+simd128',
      'Disassembly or WAT output containing f64x2 instructions in the hot loop bodies',
      'Compiler or build output showing the SIMD-enabled artifact under test',
    ]),
    passRule: 'The SIMD candidate must provide disassembly evidence of vectorized double-precision operations in the benchmarked kernels, with no scalar-only hot-loop fallback in the inspected artifact.',
  }),
  throughputCriteria: Object.freeze({
    baselineMethod: 'Run the current JS path on the same host, with the same datasets, same warmup, same iteration count, and the same timer source',
    warmupRuns: 5,
    measurementRuns: 7,
    statistic: 'median throughput',
    speedupThresholds: Object.freeze({
      rollingWindow: 1.15,
      batchEvaluation: 1.15,
      indicatorBundle: 1.10,
    }),
    noRegressionRule: 'The SIMD/WASM candidate must beat the JS baseline on throughput for the benchmark suite without exceeding the precision tolerances or breaking any bit-for-bit expectations, with no precision regressions',
  }),
  reporting: Object.freeze([
    'Report per-case throughput, median speedup, and p95 spread',
    'Report max absolute error, max relative error, and any bit-for-bit mismatches',
    'Attach the disassembly snippet or compiler remark that proves f64x2 vectorization',
    'Fail the gate if any benchmark regresses precision or if the measured speedup is below threshold',
  ]),
});

module.exports = Object.freeze({
  VALIDATION_PLAN,
});
