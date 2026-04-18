# Dreamer WASM boundary

The Dreamer ↔ numeric-core seam is now explicit and stable.

Authoritative files:
- `wasm/dreamer_kernel_contract.js` — signed-off hot-path scope, call boundary, buffer ownership, error semantics, and fallback policy
- `wasm/dreamer_kernel_adapter.js` — orchestration-only adapter that validates, stages typed buffers, selects the backend, and never performs hot-path math
- `wasm/dreamer_numeric_core.js` — JS Float64 numeric core used as the reference fallback backend with the same call shape as the WASM backend
- `wasm/dreamer_handoff_bridge.js` — Dreamer object-graph bridge; JS extracts fields from `portfolioSummary` and history objects, then stops touching math
- `wasm/src/lib.rs` — WASM exports, including explicit `_into` entry points for stable caller-owned output buffers

Hot-path computations scoped to the numeric core:
- ring-buffer state update for per-series price/volume slabs
- rolling-window mean, variance, VWAP, latest-price, momentum, and mean-volume transforms
- batched feature extraction with fixed `FEATURE_COUNT` stride writes
- portfolio deviation / crash-protection / harvest / rebalance scans
- defect scan trigger-hit and drawdown math
- regime ROI / mean / volatility / classification math

JS responsibilities only:
- map symbols to series IDs
- stage `Float64Array` / `Uint32Array` buffers
- select backend mode (`wasm-simd`, `wasm-scalar`, `js-float64-core`)
- read results after the numeric core writes them
- handle retries, logging, persistence, and other object-heavy orchestration work

JS must not perform:
- rolling reductions or ring arithmetic after staging numeric inputs
- portfolio deviation or crash-protection math
- defect-scan drawdown math
- regime variance / volatility math

Stable call boundary summary:
- `ingest_tick(seriesId, price, volume)`
- `ingest_batch(seriesId, prices, volumes, priceOffset, volumeOffset, count)`
- `compute_features_into(seriesId, windowSize, out, outOffset)`
- `compute_features_batch_into(seriesIds, windowSizes, out, outOffset, count)`
- `compute_features_batch_fixed_window_into(seriesIds, windowSize, out, outOffset, count)`
- `compute_portfolio_into(values, baselines, params, aggregateOut, deviationsOut, harvestOut, rebalanceOut)`
- `scan_defects_into(prices, rebalanceTrigger, crashThreshold, out)`
- `compute_regime_into(history, currentPrice, startPrice, out)`

Buffer ownership:
- adapter owns the long-lived arena and scratch memory
- when a WASM backend exports its own `WebAssembly.Memory`, the adapter migrates the arena into that memory once and rebinds all typed views there
- caller owns input and output typed arrays
- numeric core mutates only documented output buffers and arena state sections
- no object-shaped records cross the hot path

Fallback order:
1. `wasm-simd`
2. `wasm-scalar`
3. `js-float64-core`

Fallback is only for backend availability or runtime traps. Invalid caller input is surfaced immediately and never falls back.
