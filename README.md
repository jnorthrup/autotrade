# autotrade

Coin graph trading with hierarchical reasoning model.

## Rotational Growth: Square Cube via Sequential Catch-Up

### Core Idea

When a layer converges, it has learned a useful feature map. Randomly expanding it
corrupts the learned manifold. Instead, rotate the converged weights by 90°, 180°,
270° to create 3 additional copies that share the original's structure but present
diverse views to the optimizer.

This is **geometric noise** — the rotations preserve statistical structure (norm,
distribution, pairwise correlations) while permuting the manifold directions. The
model gets capacity to learn new interactions without discarding prior convergence.

### Why This Works

A converged layer W has learned an embedding manifold M in activation space.
A 90° rotation R(W) maps the same manifold into an orthogonal subspace.

- W captures direction D1
- R90(W) captures direction D2 (orthogonal to D1)
- R180(W) captures direction D3 (orthogonal to both)
- R270(W) captures direction D4 (orthogonal to all three)

The rotated copies aren't random — they're **orthogonal noise with preserved
magnitudes**. Training on the expanded layer learns to recombine these views.

### Invariant: Never 2, Always 4, Only Squares, Max 2 Powers

Growth factor is ALWAYS 4×. There is no 2×2×2. There is no 8×8×8. The
dimensions are always 4^k:

```
Valid powers: 4, 16, 64, 256
```

**At most 2 different powers of 4 in use at any time.** Dimensions can differ:

```
4×4×16  — valid (two powers: 4, 16)
4×16×16 — valid (two powers: 4, 16)
4×16×64 — INVALID (three powers: 4, 16, 64)
```

Growth cycle: one dimension expands by 4× at a time, any order:

```
Phase 0:  4×4         (square)
Phase 1:  4×16        (z leads, ×4)
Phase 2:  16×16       (h catches up, ×4)
Phase 3:  16×64       (z leads, ×4)
Phase 4:  64×64       (h catches up, ×4)
```

### Rotation Semantics

For a weight matrix W ∈ ℝ^{h_out × h_in} (PyTorch convention):

```python
def rotate_90(W):
    """90° counter-clockwise rotation in weight space."""
    return torch.rot90(W, k=1, dims=(-2, -1))

def rotate_180(W):
    """180° rotation."""
    return torch.rot90(W, k=2, dims=(-2, -1))

def rotate_270(W):
    """270° rotation (90° clockwise)."""
    return torch.rot90(W, k=3, dims=(-2, -1))
```

These are exact linear isometries: ‖R(W)‖₂ = ‖W‖₂, det(R) = ±1.

### The Quadrant Layout

When a converged layer is expanded, the rotated copies fill a 2×2
quadrant grid:

```
┌────────────────┬─────────────────┐
│                │                 │
│   W  (0°)      │  R180(W)        │
│   TOP LEFT     │  TOP RIGHT      │
│                │                 │
├────────────────┼─────────────────┤
│                │                 │
│  R90(W)        │  R270(W)        │
│  BOTTOM LEFT   │  BOTTOM RIGHT   │
│                │                 │
└────────────────┴─────────────────┘
```

- **Top-left: 0° (original)** — This is the converged layer. Preserved
  exactly. Never overwritten. This quadrant carries forward across ALL
  future generations as the permanent foundation.
- **Top-right: 180°** — The converged weights flipped upside-down.
- **Bottom-left: 90°** — The converged weights rotated counter-clockwise.
- **Bottom-right: 270°** — The converged weights rotated clockwise.

### Quadrant Preservation Across Generations

The top-left quadrant is NEVER overwritten, across ALL catch-up phases:

```
Generation 0:  4×4×4 converged
               The 4×4 block W is the "original"

Generation 1:  4×4×16
               W2 becomes 4×16:
               ┌────────┬────────┐
               │ W (0°) │R180(W) │  ← same 4×4 original, now in 2 quadrants
               └────────┴────────┘
               The original 0° block is still there, top-left.

Generation 2:  4×16×16
               W1 becomes 16×16:
               ┌────────┬────────┐
               │ W (0°) │R180(W) │  ← original 4×4 still top-left
               ├────────┼────────┤
               │R90(W)  │R270(W) │
               └────────┴────────┘
               The original 0° block is STILL there, top-left.

Generation 3:  16×16×16
               Now z catches up too, but the 0° quadrant is still the
               original 4×4 block in the top-left of every rotated copy.
```

The original converged weights form an **eternal anchor** — every future
expansion is a rotation of the foundation, never a replacement. The
top-left 0° quadrant is a permanent artifact across all generations.

### Concrete Example: 4×4×4 → 4×4×16

```python
# Converged state: W2 is 4×4
W2 = model.encoder[2].weight  # shape: (4, 4)

# Lead z-dimension: W2 becomes 4×16
W2_new = torch.cat([
    W2,              # 0° (original), preserves convergence
    rotate_180(W2),  # 180° rotated
    rotate_90(W2),   # 90° rotated  
    rotate_270(W2),  # 270° rotated
], dim=1)            # cat along columns (z dimension)

# W2_new is now (4, 16) = 4 rows × 4 quads of 4 cols
# ┌──────┬──────┬──────┬──────┐
# │  0°  │ 180° │  90° │ 270° │  each 4×4 block
# └──────┴──────┴──────┴──────┘
# The 0° block is identical to the original converged W2.
```

### Concrete Example: 4×4×16 → 4×16×16

```python
# Converged state: W1 is 4×x_p, W2 is 4×16
W1 = model.encoder[0].weight  # shape: (4, x_p)

# Catch up h-dimension: W1 becomes 16×x_p
W1_new = torch.cat([
    W1,              # 0° (original), preserves convergence
    rotate_180(W1),  # 180° rotated
    rotate_90(W1),   # 90° rotated
    rotate_270(W1),  # 270° rotated
], dim=0)            # cat along rows (h dimension)

# W2 (4×16) must also grow rows to match new h=16:
W2 = model.encoder[2].weight  # shape: (4, 16)

# Expand rows: take each 4×4 quadrant of W2 and rotate it
# W2 is already 4×16, we need 16×16
W2_quads = W2.split(4, dim=1)  # four 4×4 blocks: [Q0, Q1, Q2, Q3]

# Each quadrant gets its own 2×2 expansion:
W2_new = torch.cat([
    torch.cat([W2_quads[0], rotate_180(W2_quads[1])], dim=1),  # top row of 2×2
    torch.cat([rotate_90(W2_quads[2]), rotate_270(W2_quads[3])], dim=1),  # bottom row
], dim=0)

# The 0° quadrant of Q0 is still the original converged W2's top-left.
# It's now nested inside the larger 16×16 matrix.
```

### Where The Seams Are

The seams are at the boundaries between quadrants:

```
When W2 goes from 4×4 → 4×16:

col 0-3   col 4-7   col 8-11  col 12-15
┌─────────┬─────────┬─────────┬─────────┐
│  orig   │ 180°    │  90°    │ 270°    │
└─────────┴─────────┴─────────┴─────────┘
         ^         ^         ^
       seam 1    seam 2    seam 3

Each seam is where the optimizer must learn to merge two
different rotational views of the same representation.
```

The 0° block (top-left) has perfect gradients — it's already converged.
The 180°, 90°, 270° blocks have structurally identical gradients but
the feature ordering is permuted. The optimizer must learn the
recombination at seam boundaries.

Over time, the model learns smooth transitions across seams. The
converged 0° block provides a backbone, and the seam regions learn
how to bridge between rotational views. This is why each step only
grows one dimension — one seam direction at a time, not four.

### Convergence Criterion (Plateau Detection)

```
loss_history = []

def is_converged(loss_val, window=100, threshold=1e-5, patience=3):
    loss_history.append(loss_val)
    if len(loss_history) < window * patience:
        return False

    for i in range(patience):
        chunk = loss_history[-window*(patience-i):]
        if len(chunk) >= window:
            recent = np.mean(chunk[-window:])
            older = np.mean(chunk[:window])
            if abs(recent - older) > threshold:
                return False

    return True  # sustained plateau → rotate-and-expand next dim
```

### What Is Preserved

| Property | Preserved By Rotation? |
|----------|----------------------|
| Weight norm | Yes (‖R(W)‖ = ‖W‖) |
| Singular value distribution | Yes (up to reordering) |
| Activation statistics | Partially (statistical moments shift) |
| Feature ordering | No (intentional diversity source) |
| Gradient flow | No (worse initially, improves with training) |
| Original 0° quadrant | **YES, permanently across all generations** |

### Bag Shift × Sequential Catch-Up

Bag shift varies problem complexity independently of model dimensions.
Each catch-up phase trains on bags matching its current capacity:

- 4×4×4: small bags
- 4×4×16: same bags, more z-representation
- 4×16×16: slightly larger bags
- 16×16×16: bigger bags, full cube
- 16×16×64: same bags, deeper z
- ...

The bag shift is a generic arity switch — it changes problem shape without
code changes. The sequential catch-up ensures the model grows one seam at
a time, never losing prior convergence.

### Rules

1. Growth factor is always 4×. Never 2×. Never 8×.
2. Only squares. h=z=depth at cube check points.
3. One dimension leads, others catch up one at a time.
4. 0° (original) stays top-left, always preserved.
5. Seams at 90° boundaries, one seam direction per growth step.
6. Converge between every step. Never skip a plateau check.
7. The original converged block is the eternal anchor.

## Active Surface

Python is the only active implementation surface in this repo.

- Ongoing work lives under `HRM/` and `ANE/`.
- The primary model path is **PyTorch** in `HRM/`, with Metal/MPS preferred on Apple Silicon.
- `ANE/` remains as research/reference material, not the default execution target.
- The Swift package, Swift sources, and Swift tests were removed.
- If old Swift behavior is still needed, recover it from git history and port it into Python instead of reviving Swift targets.
