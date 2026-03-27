# GraphShowdown Swift Port

## Overview

Complete Swift port of `graph_showdown.py` with full autoresearch capability and square cube progression.

## Files

- `GraphShowdown.swift` - Main autoresearch implementation
- `CoinGraph.swift` - Trading graph with trial graph support
- `HRMModel.swift` - Hierarchical Reasoning Model
- `main.swift` - Entry point

## Key Features Transcribed

### Square Cube Progression

- Starts with `h_dim=4, z_dim=4`
- Grows via 4× rotational expansion
- Cycles through dimensions: h → H → L
- Enforces square invariant: `h_dim == z_dim` (validated at init)

### Convergence Detection

- Plateau window: 100 updates
- Threshold: 1e-5
- Patience: 3 consecutive windows

### Autoresearch Loop

1. Curriculum learning (pairs × window progress from simple to complex)
2. Hyperparameter sampling (log-uniform for learning rate, categorical for depths)
3. Training with loss tracking
4. Growth decision based on convergence

### Trial Graph Generation

- Selects related pairs via graph traversal
- Creates temporally-aligned windows
- Maintains graph topology

## Usage

### Standard Training

```bash
swift run Autotrade \
  --exchange coinbase \
  --start-bar 0 \
  --end-bar 10000 \
  --h-dim 4 \
  --z-dim 4 \
  --lr 0.001
```

### Autoresearch Mode

```bash
swift run Autotrade --autoresearch
```

## Differences from Python

1. **Type Safety**: Swift provides compile-time type checking
2. **Memory Safety**: ARC instead of manual memory management
3. **Concurrency**: `async/await` instead of threading
4. **Error Handling**: Result types instead of exceptions

## Verification

All core algorithms ported identically:
- Fisheye compression
- Convergence detection
- Pair selection (graph traversal)
- Growth cycle logic
- Hyperparameter sampling

## Build Status

✅ Compiles without errors
✅ Square dimension invariant enforced
✅ Trial graph generation working
✅ Autoresearch loop complete

---

*Port completed March 2026. Enforces AGENTS.md square dimension invariants.*
