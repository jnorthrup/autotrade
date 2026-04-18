# Dreamer WASM Interface Specification v1.0.0

## Purpose

This specification defines the exact compute boundary where Dreamer hands off numeric work to the WASM layer.

The boundary is stable because:
- JS performs orchestration, object traversal, staging, backend selection, and result reading only
- the numeric core performs all precision-sensitive math in every backend mode
- all hot-path calls use fixed typed-array shapes and documented buffer ownership

## Signed-off hot-path scope

Owned by the numeric core:
1. ring-buffer state update for price/volume series
2. rolling-window feature transforms
3. batched feature extraction with fixed output stride
4. portfolio deviation / crash-protection / harvest / rebalance scans
5. defect scan trigger-hit and drawdown math
6. regime ROI / mean / volatility / classification math

Excluded from JS orchestration:
- per-tick object arithmetic
- history `reduce` calls for variance / volatility
- portfolio deviation and crash-protection threshold math
- defect scan drawdown loops
- any hot-path numeric fallback outside the numeric core backend

## Buffer model

Long-lived arena layout:
- control header: 6 f64 slots
- per-series metadata: 2 f64 slots per series
- price slabs: `maxSeries * samplesPerSeries`
- volume slabs: `maxSeries * samplesPerSeries`
- feature slabs: `maxSeries * FEATURE_COUNT`

Alignment:
- 16-byte aligned base pointer
- `Float64Array` / `f64` only on the hot path

Ownership:
- adapter owns arena and scratch buffers
- if the WASM backend exports linear memory, the adapter migrates the arena into that memory once before dispatch and rebinds all typed views there
- caller owns typed input and output arrays
- numeric core mutates only documented arena/output regions
- scratch buffers are transport-only and never become external state

## Typed call boundary

### `ingest_tick`
Inputs:
- `u32 seriesId`
- `f64 price`
- `f64 volume`

State inputs:
- series metadata in the arena
- price slab for the target series
- volume slab for the target series

Outputs:
- updated series metadata
- updated control header (`lastSeriesId`, `lastPrice`, `lastVolume`)
- return value: current `sampleCount`

### `ingest_batch`
Inputs:
- `u32 seriesId`
- `Float64Array prices[count]`
- `Float64Array volumes[count]`
- `u32 priceOffset`
- `u32 volumeOffset`
- `u32 count`

Batching rule:
- one contiguous batch per series
- contract guarantees at most one ring wrap per call

Outputs:
- same mutated arena state as `ingest_tick`
- return value: current `sampleCount`

### `compute_features_into`
Inputs:
- `u32 seriesId`
- `u32 windowSize`
- `Float64Array out[FEATURE_COUNT]`
- `u32 outOffset`

State inputs:
- arena metadata
- arena price slab
- arena volume slab

Outputs:
- `out = [mean_price, price_variance, vwap, latest_price, price_momentum, mean_volume]`

### `compute_features_batch_into`
Inputs:
- `Uint32Array seriesIds[count]`
- `Uint32Array windowSizes[count]`
- `Float64Array out[count * FEATURE_COUNT]`
- `u32 outOffset`
- `u32 count`

Batching unit:
- one `{seriesId, windowSize}` row per batch element

Outputs:
- fixed `FEATURE_COUNT` stride write per row

### `compute_features_batch_fixed_window_into`
Inputs:
- `Uint32Array seriesIds[count]`
- `u32 windowSize`
- `Float64Array out[count * FEATURE_COUNT]`
- `u32 outOffset`
- `u32 count`

Outputs:
- fixed `FEATURE_COUNT` stride write per row using a shared window size

### `compute_portfolio_into`
Inputs:
- `Float64Array values[assetCount]`
- `Float64Array baselines[assetCount]`
- scalar params: `cashBalance`, `harvestTrigger`, `rebalanceTrigger`, `cpTriggerAssetPercent`, `cpTriggerMinNegativeDev`
- `Float64Array aggregateOut[5]`
- `Float64Array deviationsOut[assetCount]`
- `Float64Array harvestOut[assetCount]`
- `Float64Array rebalanceOut[assetCount]`

Outputs:
- aggregate: `[deviation_percent, crash_active, declining_count, managed_baseline, baseline_diff]`
- vector outputs: per-asset deviations, harvest flags, rebalance flags

### `scan_defects_into`
Inputs:
- `Float64Array prices[priceCount]`
- `f64 rebalanceTrigger`
- `f64 crashThreshold`
- `Float64Array out[3]`

Outputs:
- `out = [is_defective, max_drawdown, trigger_hits]`

### `compute_regime_into`
Inputs:
- `Float64Array history[historyLen]`
- `f64 currentPrice`
- `f64 startPrice`
- `Float64Array out[4]`

Outputs:
- `out = [regime_code, roi, volatility, mean]`

## JS stop points

JS must stop touching hot-path math at these points:
- after extracting `seriesId`, `price`, and `volume` scalars from Dreamer objects
- after staging `Uint32Array seriesIds/windowSizes` for batch feature calls
- after staging `Float64Array values/baselines/history/prices` for portfolio, regime, and defect scans
- after selecting the output buffer for a numeric call

From that point forward, the numeric core owns all arithmetic until the call returns.

## Error semantics

- invalid argument/type/range: throw immediately; do not fall back
- missing export or ABI mismatch: activate the next backend mode
- WASM trap: surface a typed backend error and replay once on `js-float64-core` when fallback is enabled
- exhausted fallback chain: fail without mutating caller-owned outputs

## Fallback order

1. `wasm-simd`
2. `wasm-scalar`
3. `js-float64-core`

All modes share the same typed-array boundary, so Dreamer orchestration code does not branch on call shape.
