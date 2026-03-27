# Agent Context for autotrade

## CRITICAL: HRM Rotary Growth Invariant

**HRM = Hierarchical Reasoning Model** — two-level reasoning (high-level node state + low-level edge predictions)

### NON-NEGOTIABLE INVARIANTS

1. **NEVER use 2× growth** — ALWAYS 4× rotation expansion
2. **Dimensions MUST be powers of 4** — valid values: 4, 16, 64, 256
3. **At most 2 different powers of 4 at any time** — h and z can differ
4. **One dimension grows at a time** — any order, no fixed sequence required
5. **0° quadrant is eternal** — top-left original weights NEVER overwritten across generations

### ROTARY RESIZE STRATEGY

When a layer converges (loss plateau detected), expand one dimension by 4× via 90°/180°/270° rotations:

```python
# Grow h (input dimension):
W1_new = torch.cat([W1, rotate_180(W1), rotate_90(W1), rotate_270(W1)], dim=0)

# Grow z (output dimension):
W2_new = torch.cat([W2, rotate_180(W2), rotate_90(W2), rotate_270(W2)], dim=1)

# Grow H or L (layers):
# Add more layers, no rotation needed
```

See `docs/rotation_growth.md` for complete mathematical justification.

### DANGEROUS MISTAKES TO AVOID

- **WRONG**: 2× growth (breaks rotational symmetry)
- **WRONG**: Growing multiple dimensions simultaneously (breaks sequential growth)
- **WRONG**: Random initialization instead of rotation (destroys learned manifold)
- **WRONG**: Non-power-of-4 dimensions (4, 16, 64, 256 are valid)

### LOADING SAVED MODELS

If `model_weights.pt` fails to load due to dimension mismatch:
1. Check if saved model uses power-of-4 dimensions
2. If current code uses non-power-of-4, FIX THE DEFAULTS to match saved model
3. DO NOT resize saved weights — current code must conform to saved model's architecture

### VERIFICATION

Before committing changes:
```bash
python -c "from hrm_model import HierarchicalReasoningModel; m = HierarchicalReasoningModel(); print(f'h={m.h_dim}, z={m.z_dim}')"
# Should print: h=4, z=4 (or other powers of 4)
```

## MUXER / PANCAKE / ANE DECISIONS

These decisions are now part of the repository contract. Do not silently drift away from them.

### HOT PATH SHAPE

- The autotrade hot path should maintain **one monotonic per-edge cursor/state object** and advance it once per bar.
- Do **not** rescan candle arrays from the beginning inside inner loops when a cursor can advance monotonically.
- Do **not** rebuild the same feature row independently for predict and train if both can consume the same muxed row.
- One bar should emit one edge feature row. Duplicate same-bar reconstruction is drift from the intended design.
- The old `pancake horizon` idea from `mp-superproject` was a historical hack. It belongs in `museum` / lineage references, not in the active sampler.

### FEATURE COMPRESSION

- The canonical compression rule remains the **mathematical curved sampler** already used by HRM (`fisheye` / curvature-based boundaries).
- For ANE-oriented paths, prefer a fixed-width **curved feature row** over ad hoc scalar packing.
- Legacy `pancake` / `horizon` code should not be revived into production without an explicit decision.
- `20` pixels over `200` candles is **not** the primary performance problem at current scales; repeated scans, duplicated work, and poor memory layout are.

### ANE / TILING / MECHANICAL SYMPATHY

- ANE input paths should use **tile-aligned packed rows** and contiguous projection storage.
- Pad feature rows to a stable tile width before ANE projection; avoid ragged row layouts in the ANE wrapper.
- Favor ring buffers and fixed-capacity rolling windows over repeated `removeFirst`, `suffix`, or shifting arrays in hot loops.
- ANE changes the economics and preferred memory layout of the training shell, but it does **not** change HRM rotary growth invariants.
- If the ANE wrapper is actually in CPU fallback mode, exit quickly to the regular HRM path instead of running a fake-ANE slow loop.

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
