# Runtime Contract

## Primary Runtime

- The active implementation path is Python/Torch.
- On this Apple Silicon machine, prefer Metal/MPS for normal training and inference.
- `HRM/README.md` explicitly treats plain PyTorch as the primary runtime and documents `HRM_DEVICE=mps`, `cuda`, or `cpu`.
- `HRM/device.py` resolves `auto` as `cuda`, then `mps`, then `cpu`. On this machine that normally means `mps`.

## ANE Rules

- `ANE/` is experimental/reference material unless the task explicitly targets ANE work.
- ANE must be truthful. If the wrapper is really in CPU fallback mode, route back to the regular HRM path instead of pretending ANE is active.
- The M3 Pro rule from `AGENTS.md` is hard: reverse-engineered ANE paths should treat `ch=512` as the only safe channel width on this machine.
- For M3 Pro tuning, chase spatial tiling, packed width, and depth instead of arbitrary channel sweeps.

## Checkpoints

- If `model_weights.pt` fails to load because of a shape mismatch, first check whether the saved model uses power-of-4 dimensions.
- Fix current defaults to match the checkpoint architecture instead of resizing the saved weights.
- Saved-model compatibility is part of repo doctrine, not optional cleanup.
