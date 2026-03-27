# Training Shell

## Root Training Flow

- `graph_showdown.py` is the active autotrade training shell.
- `run_training()` advances one bar at a time:
  - `graph.update(bar_idx)`
  - `model.predict(...)` after enough history exists
  - `model.update(...)` after the prediction horizon has elapsed
- `run_autoresearch()` samples pair bags and time windows, trains one trial, logs to DuckDB, and applies plateau-driven growth.

## Curriculum And Search

- Pair bags come from graph adjacency and random seeded traversal.
- Time windows are random bounded slices of `common_timestamps`.
- Trial knobs currently include `lr`, `y_depth`, `x_pixels`, `curvature`, `prediction_depth`, `H_layers`, and `L_layers`.
- Plateau detection is `_is_converged()` with a fixed window, threshold, and patience.
- Growth order in the shell is `h -> H -> L`.

## Ownership Boundary

- Root autotrade training behavior belongs in `graph_showdown.py` plus the root model wrapper.
- `HRM/pretrain.py` and `HRM/evaluate.py` remain useful donor or benchmark flows, but they are not the current autotrade hot path.

## Use This Reference When

- Adjusting autoresearch or curriculum logic
- Reconciling checkpoint growth behavior with plateau detection
- Deciding whether a change belongs in the root wrapper or the donor `HRM/` tree
