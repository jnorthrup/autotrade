# Conductor Tracks

## Track 1: Fix growth cycle divergence between graph_showdown.py and model

**Status:** CLOSED
**Priority:** HIGH
**Owner:** codex

### Changes made
- Added `_next_square_cube_size()` — returns next 4^k size or None at ceiling
- Added `_validate_square_cube_state()` — enforces powers of 4 with at most 2 distinct values
- Added `_apply_growth_step()` — single 4× growth step, delegates to model.grow()
- Changed initial state from (4,1,1) to (4,4,4) to satisfy invariant
- Replaced manual growth logic with `_apply_growth_step()` call
- Removed `H_layers * 2` / `H_layers + 1` violations

### Verified
- Syntax: python3 -m py_compile graph_showdown.py hrm_model.py
- Growth sequence: [(4,4,4), (16,4,4), (16,16,4), (16,16,16), (64,16,16), (64,64,16), (64,64,64)]
- Training loop: avg_loss=1.135 on 50 bars

## Track 2: End-to-end smoke test with small dataset

**Status:** CLOSED
**Priority:** MEDIUM
**Owner:** master

### Changes made
- Created tests/smoke_test.py
- Loads graph (skip_fetch, lookback_days=30, min_partners=3, max_partners=5)
- Creates model (h_dim=4, z_dim=4)
- Trains on 50 bars, asserts n_updates > 0 and avg_loss < 3.0
- Saves/loads checkpoint, verifies dimensions match

### Verified
```bash
python3 tests/smoke_test.py
# === PASS ===
# avg_loss=2.058732, n_updates=48
# checkpoint round-trip OK
```

## Track 3: Move HRM model to Metal/MPS GPU

**Status:** CLOSED
**Priority:** HIGH
**Owner:** subagent

### Changes made
- Added `get_device()` function at module level
- `_build_model()`: `self._device = get_device()`, `.to(self._device)`
- `__init__()`: `self._device = get_device()` default
- `update()`: all `torch.tensor()` calls include `device=self._device`
- `predict()`: same tensor device fix
- `load()`: `map_location=self._device` in both `torch.load()` calls
- Fixed bfloat16 MPS bug: changed `H_init`, `L_init`, `z_H`, `z_L` from `torch.bfloat16` to `torch.float32`

### Verified
- Device: `mps:0` ✅
- Smoke test: PASS (avg_loss=2.06) ✅

## Track 4: Stochastic bag sampling with volatility filtering

**Status:** OPEN
**Priority:** HIGH
**Owner:** unassigned

### Problem
Current training uses a fixed bag (bag.json) and fixed time windows. Model memorizes specific pairs/periods instead of generalizing.

### Required: stochastic bag sampler in graph_showdown.py
1. Load all available Binance pairs from candle_cache
2. Filter out low-volatility fiat/stablecoin pairs (keep crypto-crypto)
3. Each training iteration: randomly sample N pairs (bag size scales with model size)
4. Each training iteration: randomly sample time window (start_bar, end_bar)
5. Bag size grows with model: 4×4×4 → small bags, 16×16×16 → larger bags
6. No explicit pathfinding — H-level reasoning discovers alpha through bipolar edge dynamics

### Volatility filter heuristic
- Compute mean absolute velocity per pair over full history
- Drop pairs below threshold (e.g., mean |velocity| < 0.001)
- Drop fiat-fiat edges (USD-USDT, EUR-USD, etc.)

### Acceptance criteria
- Training loop uses stochastic bags, not fixed bag.json
- Loss averaged across multiple bag samples per epoch
- Model converges on unseen pairs (generalization test)

## Track 5: Daytrade bag fine-tuning + periodic retrain

**Status:** OPEN
**Priority:** HIGH
**Owner:** unassigned

### Two-phase training model
1. **Pre-train** (graph_showdown.py): stochastic bags, many pairs, long history. Produces `model_weights_pretrained.pt`
2. **Fine-tune** (trading harness): load pretrained checkpoint, train on fixed daytrade bag (bag.json pairs), short recent window. Produces `model_weights_daytrade.pt`
3. **Periodic re-fine-tune**: re-run phase 2 on schedule (daily/weekly) to adapt to regime shifts

### Required
- `--fine-tune` flag in graph_showdown.py or separate `finetune.py` script
- Loads pretrained checkpoint, freezes growth (no 4× expansion), trains on fixed bag
- Adjustable learning rate (lower for fine-tune than pre-train)
- Checkpoint naming: `model_weights_pretrained.pt` vs `model_weights_daytrade.pt`
- Timestamp metadata in checkpoint for staleness tracking

### Acceptance criteria
- Can run: `python3 finetune.py --pretrained model_weights_pretrained.pt --bag bag.json --lr 0.0001`
- Fine-tuned model outperforms pretrained model on daytrade bag pairs
- Checkpoint includes timestamp for staleness detection
