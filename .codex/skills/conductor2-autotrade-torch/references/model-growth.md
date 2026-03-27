# Model And Growth

## Root Model Surface

- `HierarchicalReasoningModel` in `hrm_model.py` is the active autotrade wrapper.
- It instantiates `HRMEdgePredictor`, registers per-edge carry state, keeps close-price history, builds fisheye features, queues predictions, scores targets, and saves checkpoints.
- `HRMEdgePredictor` is a two-level H/L reasoning stack with:
  - `FisheyeEmbedding`
  - `H_level` and `L_level` reasoning modules
  - carry tensors `z_H` and `z_L`
  - task heads `fraction`, `PTT`, and `STOP`

## Rotary Growth Invariants

- Growth is always 4x, never 2x.
- Valid hidden dimensions are the power-of-4 set: `4`, `16`, `64`, `256`.
- One dimension grows at a time.
- At most two powers of 4 may be active at once for `h` and `z`.
- The top-left 0-degree block is eternal and must never be overwritten.

## Current Root Behavior

- `h_dim` is the actual hidden size used to instantiate the root predictor.
- `z_dim` is stored in wrapper state and checkpoints, but the current root predictor is still a single-hidden-size path. Do not assume independent `h` and `z` growth exists in the root implementation yet.
- `grow('h')` advances hidden size through `SQUARE_CUBE_SIZES = [4, 16, 64, 256]`.
- `grow('H')` and `grow('L')` expand reasoning depth by multiplying the current layer stack by 4.
- Checkpoint saves include `x_pixels`, `y_depth`, `curvature`, `h_dim`, `z_dim`, `prediction_depth`, `H_layers`, `L_layers`, `H_cycles`, and `L_cycles`.

## Donor Context

- `HRM/models/hrm/hrm_act_v1.py` is still the best donor reference for ACT-style H/L carry semantics.
- Use donor code to absorb working PyTorch patterns into the active root surface, not to preserve stale architecture assumptions.
