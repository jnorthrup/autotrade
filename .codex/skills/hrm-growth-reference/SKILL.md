---
name: hrm-growth-reference
description: Repo-local reference for autotrade HRM growth, GraphShowdown plateau growth, rotary resizing, checkpoint invariants, and the vocabulary around experience-preserving telescoping. Use when changing Sources/AutotradeHRM/HRMModel.swift, Sources/Autotrade/GraphShowdown.swift, ANE-backed HRM training, or when comparing the Swift port against museum/python/hrm_model.py and ../HRM.
---

# HRM Growth Reference

Use this skill when working on the autotrade hierarchical reasoning model, its growth rules, or its GraphShowdown training shell.

## Read Order

1. Read [`AGENTS.md`](../../../AGENTS.md) first. The rotary growth invariant there is authoritative.
2. Read [`docs/rotation_growth.md`](../../../docs/rotation_growth.md) for the geometry and the preserved 0-degree anchor.
3. Read [`Sources/AutotradeHRM/HRMModel.swift`](../../../Sources/AutotradeHRM/HRMModel.swift) for the current Swift backbone and checkpoint contract.
4. Read [`Sources/Autotrade/GraphShowdown.swift`](../../../Sources/Autotrade/GraphShowdown.swift) for plateau growth, bag growth, and autoresearch control flow.
5. When parity matters, read [`museum/python/hrm_model.py`](../../../museum/python/hrm_model.py) and inspect the sibling `../HRM` checkout.

## Vocabulary

- `backbone`: the hierarchical H/L reasoning core, including `H_level`, `L_level`, `z_H`, `z_L`, and the H/L cycle schedule.
- `task heads`: the supervised outputs `fraction`, `PTT`, and `STOP`. These are not the old codec predictor.
- `growth policy`: the geometry rule used for each tensor growth step.
- `experience`: the preserved 0-degree anchor block. In plain language: top-left quad is experience.
- `weight forwarded old experience`: preserve former convergence as an addressable sub-block instead of replacing it with random noise.
- `smooth bricks`: grow one dimension at a time so the optimizer only has to integrate one new seam direction at once.

## Invariants

- Dimensions must stay in the power-of-4 set: `4`, `16`, `64`, `256`.
- One dimension grows at a time.
- At most two different powers of 4 may be active at once.
- The top-left 0-degree block is prior experience and is never overwritten.
- For axis-only growth, the fallback layout is `0,180,180,0`.
- `bridge` is a historical reference and compatibility mode only. It is not the default production growth rule.

## Training Shell

- Plateau growth means convergence drives model growth.
- bags are stochastic Dykstra picks and stochastic candle spans.
- Bag complexity should rise with current model capacity, not raw phase count.
- ANE changes training economics, not preservation math.
- Treat one per-edge cursor and one compressed row per bar as the canonical hot-path shape.
- The old `mp-superproject` `pancake horizon` is a historical hack and belongs in `museum` / lineage references, not in the active sampler.
- For ANE-era ports, adapt the old design to tile-aligned packed rows and contiguous projection storage.
- Keep the existing curvature-based / fisheye mathematical sampler as the default compression rule unless there is an explicit reason to replace it.
- Use `horizon` only as a historical reference for lineage or museum parity, not as the default substitute for the curved sampler.
- Do not confuse `20 x 200` feature size with the main slowdown. The first fixes are monotonic cursors,
  shared muxed rows, fallback discipline, and memory layout.

## Implementation Guidance

- Prefer explicit carry types and edge-level APIs over opaque predictor blobs.
- Preserve the task heads even when the backbone changes.
- Save checkpoints with growth metadata so resume behavior is shape-safe and vocabulary-stable.
- Treat random restart as an explicit baseline or fallback, never the default growth operator.
- In hot loops, prefer ring buffers and fixed-capacity windows over array shifting.
- If an ANE wrapper is actually in CPU fallback, hand back to the regular HRM shell instead of simulating ANE slowly.
