---
name: conductor2-autotrade-torch
description: Repo-local Conductor2 overlay for /Users/jim/work/autotrade. Use when working on autotrade Python/Torch code, graph_showdown.py, hrm_model.py, coin_graph.py, the embedded HRM/ donor repo, runtime/backend routing, rotary growth invariants, checkpoint compatibility, or when you need the project-specific read order before editing.
---

# Conductor2 Autotrade Torch

Use this skill for non-trivial work in `autotrade` when repo doctrine matters more than generic model knowledge.

## First Principles

- `AGENTS.md` is authoritative for rotary growth, backend choice, hot-path shape, and truthful ANE behavior.
- The active implementation surface is Python/Torch. Do not drift back to Swift or private-ANE doctrine unless the task explicitly asks for it.
- The root files `hrm_model.py`, `graph_showdown.py`, `coin_graph.py`, and `layers_fallback.py` are the active autotrade shell to inspect first.
- `HRM/` is the donor/reference PyTorch tree. Use it to absorb working patterns into Python, not to justify ignoring repo-local contracts.
- Do not trust memory for this repo. Load the relevant reference file below, then open the target code.

## Read Order

1. Read [`AGENTS.md`](../../../AGENTS.md).
2. Read [`README.md`](../../../README.md).
3. Read one or more reference files below.
4. Open the exact source files you will touch.

## Reference Routing

- Project map and file ownership: [`references/project-map.md`](references/project-map.md)
- Runtime and backend contract: [`references/runtime-contract.md`](references/runtime-contract.md)
- HRM model shape, checkpoint, and growth rules: [`references/model-growth.md`](references/model-growth.md)
- Training shell, bags, and autoresearch loop: [`references/training-shell.md`](references/training-shell.md)
- Hot-path contract and current code mismatches: [`references/hot-path-and-gaps.md`](references/hot-path-and-gaps.md)
- Deep rotary-growth geometry and older parity vocabulary: [`../hrm-growth-reference/SKILL.md`](../hrm-growth-reference/SKILL.md)

## Working Rules

- Preserve the 4x rotary growth invariant. If current code disagrees, the code is wrong, not the invariant.
- Prefer MPS on this Apple Silicon machine. ANE paths must be truthful and fail closed when unsupported.
- One bar should yield one compressed edge row. Avoid same-bar feature reconstruction drift.
- Saved-model compatibility beats convenience. Fix defaults to match power-of-4 checkpoints instead of resizing checkpoints.
- When the root wrapper and `HRM/` diverge, name the intended owner before editing.
