# Validation plan for numerical equivalence and throughput

This plan defines how the SIMD/WASM path is validated against the current JS Float64 path.
It is written to satisfy three gates:

1. benchmark coverage for representative rolling-window and indicator workloads
2. numerical equivalence with explicit bit-for-bit and tolerance rules
3. measurable throughput improvement with no precision regressions

## Benchmark cases

The benchmark suite must include all of the following cases:

- Rolling-window steady state
  - single-series feature extraction over contiguous ring-buffer spans
  - exercises `writeWindowFeatures`
  - data shape: 1 series, 16k ticks, 256-sample ring buffer, windows 5/20/64/128

- Rolling-window wraparound
  - single-series reduction after the write index wraps
  - exercises the tail-span + head-span path
  - data shape: 1 series, 16k ticks, 128-sample ring buffer, windows 20/64/96

- Batch series/window scan
  - many series and many window sizes in one contiguous batch
  - exercises `evaluateBatch`
  - data shape: 256 series, 65k evaluated rows, windows 5/20/64/128

- Indicator bundle
  - representative indicator-style rolling transforms from the current JS path
  - includes SMA, EMA, ATR, RSI, MACD, VWAP, and momentum-style scans
  - data shape: 32 series, 50k bars, OHLCV-style price and volume traces, windows 10/12/14/20/26/60

## Numerical equivalence rules

The validation gate distinguishes between exact-copy behavior and reduction behavior.

Bit-for-bit expectations:

- Float64 ingress and egress through the adapter and arena views must round-trip exactly
- Direct copy outputs, including `latest_price`, must match the JS baseline bit-for-bit
- NaN and Infinity classifications must match the JS baseline exactly

Tolerance rules for reduction outputs:

- absolute error <= 1e-12
- relative error <= 1e-12
- ULP drift <= 1

Reduction outputs include mean, variance, VWAP, momentum, and indicator aggregates.

Comparison rules:

- use bit-for-bit comparison for direct copies and boundary round-trips
- use absolute/relative tolerance for reductions
- treat NaN as matching only when the baseline is also NaN in the same slot
- reject any sign flip, category mismatch, or tolerance miss as a precision regression

## Auto-vectorization evidence

The SIMD build must provide evidence that the hot loop is actually vectorized:

- compile with `-C target-feature=+simd128`
- inspect disassembly or WAT output for `f64x2` instructions in the hot loop bodies
- keep the build artifact and the inspected snippet in the validation report

Pass condition:

- the inspected SIMD artifact shows double-precision vector operations in the benchmarked kernels
- no scalar-only hot-loop fallback is present in the inspected artifact

## Throughput success criteria

Benchmark methodology:

- compare against the current JS path on the same host
- use the same datasets, same warmup, same iteration count, and the same timer source
- run 5 warmup passes and 7 measured passes
- report median throughput and p95 spread

Success thresholds:

- rolling-window benchmarks: at least 15% faster than the JS baseline
- batch evaluation benchmarks: at least 15% faster than the JS baseline
- indicator bundle benchmarks: at least 10% faster than the JS baseline

Failure conditions:

- any benchmark misses the numerical tolerances above
- any bit-for-bit expectation is violated
- the measured throughput improvement falls below the thresholds above
- the SIMD evidence does not show `f64x2` operations in the hot loops

## Required report contents

Each validation run must record:

- benchmark name and dataset shape
- baseline and candidate median throughput
- speedup factor
- max absolute error and max relative error
- any bit-for-bit mismatches
- the disassembly or compiler evidence used to confirm auto-vectorization
