# Hot Path And Gaps

## Intended Hot-Path Contract

- One monotonic per-edge cursor/state object should advance once per bar.
- One bar should emit one compressed curved feature row per edge.
- Predict and train should consume the same muxed row instead of rebuilding it independently.
- The canonical compression rule is the curved sampler already used by HRM, not revived pancake or horizon hacks.
- If ANE is used, prefer tile-aligned packed rows and contiguous projection storage.

## Current Root Observations

- `coin_graph.py` aligns timestamps by union, not intersection, and handles missing bars per edge at update time.
- The root shell now batches per-bar predict/update tensor work, but fisheye row construction and `coin_graph.py` still spend real time in Python/Pandas. Profile before claiming the device is the bottleneck.
- Readiness is now driven by observed per-edge candle counts rather than global `bar_idx`, so any future refactor must preserve per-edge maturity semantics.
- `graph_showdown.py` keeps the general training shell on `auto`, while day-trading / fine-tune paths should explicitly choose `cpu` unless there is a measured reason not to.
- `HRMEdgePredictor.grow_hidden_size()` still prepares prediction-head tensors as if the heads were square hidden-to-hidden projections, even though the heads are defined as `Linear(hidden_size, 1)`. Treat hidden-size growth as unsafe until inspected and fixed.

## How To Use This File

- Treat the items above as active reconciliation points, not doctrine.
- Before trusting existing behavior, check whether your change touches one of these mismatches.
- If you fix one, update this reference so the next turn does not rediscover it from scratch.
