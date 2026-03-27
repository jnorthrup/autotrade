# Project Map

## Authority Order

1. `AGENTS.md`
2. Root Python code and repo-local docs
3. `HRM/` donor code
4. Historical ANE or museum lineage

## Active Root Surfaces

- `hrm_model.py`: active autotrade wrapper. Owns `HRMEdgePredictor`, the `HierarchicalReasoningModel` wrapper, fisheye sampling, per-edge carry, checkpoint save/load, and growth entrypoints.
- `graph_showdown.py`: active training shell and CLI. Owns bag selection, plateau detection, autoresearch growth order, experiment logging, and bar-by-bar training flow.
- `coin_graph.py`: market graph discovery and edge/node state updates. Owns pair loading, candle alignment, velocity/accel/PTT/STOP updates, and node heights.
- `layers_fallback.py`: Torch fallback attention, RoPE, RMSNorm, and SwiGLU used by the root wrapper without depending on FlashAttention.
- `config.py` and `candle_cache.py`: environment, database, and candle ingestion surfaces.

## Embedded Donor Surfaces

- `HRM/README.md`: current Torch-first runtime notes and benchmark-style training instructions.
- `HRM/device.py`: device routing helpers for `auto`, `mps`, `cuda`, and `cpu`.
- `HRM/models/hrm/hrm_act_v1.py`: donor ACT HRM backbone and carry semantics.
- `HRM/models/layers.py`, `HRM/models/common.py`, `HRM/models/losses.py`: donor layer and loss implementations.
- `HRM/pretrain.py` and `HRM/evaluate.py`: donor benchmark training and evaluation entrypoints, not the active autotrade training shell.

## Default Inspection Order

- Open the root file that currently owns the behavior first.
- Open `HRM/` only after you know whether you are borrowing a pattern, reconciling divergence, or moving logic into the root Python path.
- Treat ANE material as experimental unless the task explicitly asks for ANE work.
