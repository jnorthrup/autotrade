# Agent Context for autotrade

## CRITICAL: HRM Rotary Growth Invariant

**HRM = Hierarchical Reasoning Model** — two-level reasoning (high-level node state + low-level edge predictions)

### NON-NEGOTIABLE INVARIANTS

1. **NEVER use 2× growth** — ALWAYS 4× rotation expansion
2. **ALWAYS retain square** — `h_dim == z_dim` at initialization and after catch-up
3. **Default dimensions MUST be square** — `h_dim=4, z_dim=4` NOT `h_dim=16, z_dim=8`
4. **Sequential catch-up only** — one dimension leads by 4×, others catch up one at a time
5. **0° quadrant is eternal** — top-left original weights NEVER overwritten across generations

### ROTARY RESIZE STRATEGY

When a layer converges (loss plateau detected), expand via 90°/180°/270° rotations:

```python
# Lead dimension (z grows first):
W2_new = torch.cat([W2, rotate_180(W2), rotate_90(W2), rotate_270(W2)], dim=1)

# Catch-up dimension (h grows second):
W1_new = torch.cat([W1, rotate_180(W1), rotate_90(W1), rotate_270(W1)], dim=0)
```

See `docs/rotation_growth.md` for complete mathematical justification.

### DANGEROUS MISTAKES TO AVOID

- **WRONG**: `h_dim=16, z_dim=8` (breaks square invariant)
- **WRONG**: 2× growth (breaks rotational symmetry)
- **WRONG**: Growing multiple dimensions simultaneously (breaks sequential catch-up)
- **WRONG**: Random initialization instead of rotation (destroys learned manifold)

### LOADING SAVED MODELS

If `model_weights.pt` fails to load due to dimension mismatch:
1. Check if saved model uses square dimensions (e.g., `h_dim=4, z_dim=4`)
2. If current code defaults to non-square, FIX THE DEFAULTS to match saved model
3. DO NOT resize saved weights — current code must conform to saved model's architecture

### VERIFICATION

Before committing changes:
```bash
python -c "from hrm_model import HierarchicalReasoningModel; m = HierarchicalReasoningModel(); print(f'h={m.h_dim}, z={m.z_dim}')"
# Should print: h=4, z=4 (or both 16, both 64, etc. — ALWAYS equal)
```

## ANE TRAINING INTEGRATION

**ANE = Apple Neural Engine** — dedicated 15.8 TFLOPS inference accelerator on M4/M3 chips

### OVERVIEW

The project includes ANE training support via reverse-engineered private APIs in `./ANE/training/`. This allows training transformer models directly on ANE hardware with ~9ms/step throughput for dim=768 models.

### FILES

- `ane_training.py` — Python interface for ANE training (export/import checkpoints)
- `ane_hrm_train.m` — Objective-C wrapper for HRM-specific ANE training
- `Makefile` — Build configuration for ANE HRM trainer
- `./ANE/training/` — Low-level ANE runtime, MIL generators, training loops

### BUILDING ANE TRAINER

```bash
make ane_hrm_train
```

### USING ANE TRAINING

```python
from ane_training import ANETrainer
from hrm_model import HierarchicalReasoningModel

model = HierarchicalReasoningModel()
model.register_edges(edges)

trainer = ANETrainer(model)
loss = trainer.train(steps=1000, batch_size=32, learning_rate=0.001)
```

### ANE TRAINING ARCHITECTURE

The ANE system uses 6 kernels per training step:
1. `kFwdAttn` — Forward attention (QKV + SDPA + output)
2. `kFwdFFN` — Forward FFN (SwiGLU)
3. `kFFNBwd` — FFN backward
4. `kSdpaBwd1` — SDPA backward part 1
5. `kSdpaBwd2` — SDPA backward part 2
6. `kQKVb` — QKV backward

See `./ANE/README.md` for complete details on ANE training implementation.

### CHECKPOINT FORMAT

ANE training uses custom binary checkpoint format (`ANECheckpointFormat`):
- Header: Magic + version + param count
- Per param: Name + shape + dtype + data

This bridges PyTorch state_dicts with ANE training code.

