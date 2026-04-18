# Double-vector Dreamer ↔ WASM boundary

This document defines the exact seam where Dreamer stops handling numbers and hands compute to the numeric core.

## Signed-off hot-path scope

The following computations are signed off for the numeric core and excluded from object-heavy orchestration code:

1. Ring-buffer state update
   - write-index wrap
   - sample-count saturation
   - contiguous SoA price and volume slab writes
2. Rolling-window feature transforms
   - mean price
   - price variance
   - VWAP
   - latest price
   - price momentum
   - mean volume
3. Batched feature extraction
   - `Uint32Array` series/window batch scans
   - fixed `FEATURE_COUNT` output stride writes
4. Portfolio vector scan
   - per-asset deviation math
   - crash-protection counters
   - harvest/rebalance flag generation
5. Defect scan
   - trigger-hit counting
   - look-ahead drawdown search
   - defect flag derivation
6. Regime statistics
   - ROI
   - mean
   - variance / volatility
   - threshold-based regime classification

These are owned by the numeric core in every backend mode: `wasm-simd`, `wasm-scalar`, and `js-float64-core`.

## What JS still does

JS owns orchestration only:
- network ingress and scheduling
- symbol-to-series lookup
- extracting fields from `portfolioSummary` and history objects
- staging `Float64Array` and `Uint32Array` inputs
- backend selection and fallback activation
- reading typed outputs after the numeric core finishes
- logging, persistence, retries, and policy decisions

JS must stop touching math immediately after it has produced typed numeric inputs for the adapter.

## Stable call boundary

Low-level boundary entry points:
- `ingest_tick(seriesId, price, volume)`
- `ingest_batch(seriesId, prices, volumes, priceOffset, volumeOffset, count)`
- `compute_features_into(seriesId, windowSize, out, outOffset)`
- `compute_features_batch_into(seriesIds, windowSizes, out, outOffset, count)`
- `compute_features_batch_fixed_window_into(seriesIds, windowSize, out, outOffset, count)`
- `compute_portfolio_into(values, baselines, params, aggregateOut, deviationsOut, harvestOut, rebalanceOut)`
- `scan_defects_into(prices, rebalanceTrigger, crashThreshold, out)`
- `compute_regime_into(history, currentPrice, startPrice, out)`

High-level Dreamer object bridge:
- `DreamerHandoffBridge.ingestPortfolioSummary(portfolioSummary)`
- `DreamerHandoffBridge.computeFeatureBatch(requests)`
- `DreamerHandoffBridge.computePortfolioFromSummary(portfolioSummary, params)`
- `DreamerHandoffBridge.computeRegimeFromHistory(historyLike, options)`
- `DreamerHandoffBridge.scanDefectsFromHistory(historyLike, options)`

The bridge is the last point where JS touches objects. The adapter boundary itself is typed-array only.

## Batching units and state handoff

Feature kernels
- state input: adapter-owned arena with control header, per-series metadata, price slabs, volume slabs, and feature slabs
- batch unit: one `{seriesId, windowSize}` row
- batch shape: `Uint32Array seriesIds[count]`, `Uint32Array windowSizes[count]`, `Float64Array out[count * FEATURE_COUNT]`
- output shape: fixed `FEATURE_COUNT` stride per row

Portfolio kernel
- inputs: `Float64Array values[assetCount]`, `Float64Array baselines[assetCount]`, scalar thresholds
- outputs:
  - aggregate `Float64Array[5]`
  - deviations `Float64Array[assetCount]`
  - harvest flags `Float64Array[assetCount]`
  - rebalance flags `Float64Array[assetCount]`

Defect kernel
- inputs: `Float64Array prices[priceCount]`, `rebalanceTrigger`, `crashThreshold`
- output: `Float64Array[3]`

Regime kernel
- inputs: `Float64Array history[historyLen]`, `currentPrice`, `startPrice`
- output: `Float64Array[4]`

## Buffer ownership

- Adapter owns the long-lived arena and scratch buffers.
- If the WASM backend exports its own `WebAssembly.Memory`, the adapter migrates the arena into that linear memory once and rebinds its typed views before hot-path calls begin.
- Caller owns all typed input and output views passed to `_into` calls.
- Numeric core may read caller inputs and write caller outputs, but it does not retain references after the call.
- Numeric core mutates only documented arena state sections.
- No object records, JSON blobs, or polymorphic maps are allowed on the hot path.

## Error semantics

- Invalid caller input: throw immediately; do not fall back.
- Missing backend export / ABI mismatch: fallback to the next backend mode.
- WASM trap: surface a typed error and replay once on `js-float64-core` when fallback is enabled.
- Exhausted fallback chain: fail without mutating caller-owned output buffers.

## Fallback behavior

Fallback order:
1. `wasm-simd`
2. `wasm-scalar`
3. `js-float64-core`

All three modes use the same typed-array boundary so Dreamer orchestration code does not change when the backend changes.
