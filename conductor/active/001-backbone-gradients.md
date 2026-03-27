# 001 Backbone Gradients

Status: closed
Owner: conductor
Objective: restore gradient flow through the root HRM backbone in `hrm_model.py` without changing the caller-facing training shell.
Bounded corpus: `hrm_model.py`
Runtime route: Codex worker via `spawn_agent`, repo `/Users/jim/work/autotrade`, branch `master`
Stop condition: one slice or first blocker

## Why This Slice Exists

- The root trainer is the active autotrade Python surface.
- `HRMEdgePredictor.forward()` wraps the H/L recurrent core in `torch.no_grad()`.
- That blocks backbone gradients during `HierarchicalReasoningModel.update()`, so training mostly collapses to the prediction heads.

## Acceptance

- Backbone forward no longer blocks gradients needed for training.
- Carry state remains detached across bars so recurrent history does not grow unbounded graphs.
- A focused gradient check shows non-`None` gradients on backbone parameters after `loss.backward()`.

## Verification

- Run a focused Python gradient check against `HierarchicalReasoningModel.update()` or `HRMEdgePredictor.forward()`.
- Inspect the resulting diff in `hrm_model.py`.

Verified by master:

- `python3 - <<'PY' ... HRMEdgePredictor(x_pixels=30, hidden_size=4, num_heads=1, H_layers=1, L_layers=1, H_cycles=2, L_cycles=2) ... loss.backward() ... PY`
- Observed non-`None` gradients on `embed.proj.weight`, `H_level.layers[0].self_attn.q_proj.weight`, and `L_level.layers[0].self_attn.q_proj.weight`.
- Verified detached carry: `carry.z_H.requires_grad == False` and `carry.z_L.requires_grad == False`.
