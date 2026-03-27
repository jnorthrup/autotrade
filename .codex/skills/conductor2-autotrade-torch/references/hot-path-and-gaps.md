# Hot Path And Gaps

## Intended Hot-Path Contract

- One monotonic per-edge cursor/state object should advance once per bar.
- One bar should emit one compressed curved feature row per edge.
- Predict and train should consume the same muxed row instead of rebuilding it independently.
- The canonical compression rule is the curved sampler already used by HRM, not revived pancake or horizon hacks.
- If ANE is used, prefer tile-aligned packed rows and contiguous projection storage.

## Current Root Observations

- `coin_graph.py` aligns timestamps by union, not intersection, and handles missing bars per edge at update time.
- `HierarchicalReasoningModel.predict()` rebuilds `_get_fisheye(edge)` twice for the same edge and bar: once for inference and once for the queued training frame.
- `HierarchicalReasoningModel.update()` later replays the queued copied row instead of consuming a shared muxed feature object.
- `HRMEdgePredictor.forward()` wraps the recurrent H/L core in `torch.no_grad()`, which blocks backbone gradients during training.
- `graph_showdown.py` still advertises `--z-dim 8`, which violates the power-of-4 rule, and the root predictor does not yet honor an independent `z` path anyway.
- `HRMEdgePredictor.grow_hidden_size()` currently prepares prediction-head tensors as if the heads were square hidden-to-hidden projections, even though the heads are defined as `Linear(hidden_size, 1)`. Treat hidden-size growth as unsafe until inspected and fixed.

## How To Use This File

- Treat the items above as active reconciliation points, not doctrine.
- Before trusting existing behavior, check whether your change touches one of these mismatches.
- If you fix one, update this reference so the next turn does not rediscover it from scratch.
