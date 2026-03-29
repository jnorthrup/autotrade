# Conductor Tracks

## Track 0: Fix DuckDB connection configuration conflict

**Status:** CLOSED
**Priority:** CRITICAL (blocked Track 4)
**Owner:** master

### Problem
DuckDB error: "Can't open a connection to same database file with a different configuration than existing connections"

Root cause: Inconsistent DuckDB connection configurations in graph_showdown.py:
- Line 340: `duckdb.connect(db_path)` (no explicit config, read-write)
- Line 152: `duckdb.connect(db_path, read_only=True)` (read-only)
- Line 184: `duckdb.connect(db_path, read_only=True)` (read-only)

DuckDB doesn't allow mixing connections with different configurations to the same database file.

### Solution implemented

**Simple fix (applied):** Removed `read_only=True` from connections at lines 152 and 184. All connections now use the same default configuration.

**Future enhancement (NOT pursued):** A singleton DuckDB connection pool server exists in `../literbike` (~/work/literbike) as `duckdb_pool` binary. This was NOT added to literbike to avoid cross-repo contamination. The pool server pattern can be revisited later if connection pooling becomes necessary at scale.

### Changes made
- **graph_showdown.py**: Removed `read_only=True` from `_list_all_binance_pairs()` and `_compute_volatility_filter()` DuckDB connections

### Verified
- Syntax: python3 -m py_compile graph_showdown.py ✅
- All DuckDB connections now use consistent configuration ✅

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

**Status:** OPEN (READY - no longer blocked)
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

## Track 6: Docker Multi-Service Deployment with Dashboard

**Status:** OPEN (IN PROGRESS)
**Priority:** HIGH
**Owner:** master (dashboard components)

### Services
1. **data-ingestion**: Live WebSocket + REST polling for candle data
2. **paper-trader**: Real-time prediction + signal generation
3. **training-pretrain**: Stochastic bag training worker
4. **training-finetune**: Fixed bag daytrade fine-tuning worker
5. **health-monitor**: Lightweight status endpoint (port 8001)
6. **dashboard-showdown**: Full metrics dashboard with drilldown (port 8000)

### Components Created
- `Dockerfile`: Python 3.11 slim, with PyTorch, DuckDB, coinbase SDK
- `docker-compose.yml`: 6 services with resource limits
- `entrypoint.sh`: Environment validation + bootstrap
- `requirements.txt`: Dependencies
- `training_worker.py`: Dual-mode training worker (pretrain/finetune)
- `health_monitor.py`: Minimal health/metrics endpoint
- `dashboard_showdown.py`: Full dashboard with:
  - Live loss charts (canvas-based)
  - 4× growth visualization
  - Phase history table with click-to-drilldown
  - Per-phase details (bag pairs, window, model state)
  - Checkpoint history
  - Manual actions (save, poll)

### Endpoints
| Endpoint | Service | Description |
|----------|---------|-------------|
| `:8000/` | dashboard-showdown | Full dashboard UI |
| `:8000/api/status` | dashboard-showdown | System status JSON |
| `:8000/api/metrics` | dashboard-showdown | Time-series metrics |
| `:8000/api/drilldown/<int>` | dashboard-showdown | Phase details |
| `:8000/api/action/<action>` | dashboard-showdown | Manual triggers |
| `:8001/health` | health-monitor | Health check |
| `:8001/metrics` | health-monitor | Quick metrics |

### Acceptance Criteria
- Single `docker-compose up` starts all services
- Dashboard shows live training progress
- Drilldown reveals bag composition per phase
- Model growth (4×) visually tracked
- Health endpoints for monitoring/alerts

## Track 7: Cron-Based Training Orchestration + Service Babysitting

**Status:** OPEN (READY)
**Priority:** HIGH
**Owner:** master

### Overview
Automated management of training lifecycle across two distinct data streams:
1. **Binance Bulk Data** → Pre-training (stochastic, long history, many pairs)
2. **Coinbase Live** → Day-trade fine-tuning (fixed bag, recent window, paper trade signals)

### Components Created

#### docker_babysitter.py
- Monitors all Docker Compose services
- Detects unhealthy/failed containers and auto-restarts
- Analyzes checkpoint files for:
  - NaN values (corrupted training)
  - Exploding loss (learning rate too high)
  - Stagnant loss (variance < 0.001 over 10 phases)
  - Stale checkpoints (>48h old)
- Removes defective checkpoints to backup directory
- Rotates training jobs when stuck

#### training_orchestrator.py
- Two-track training management:
  - **Pretrain**: Continuous stochastic bag training on Binance bulk data
  - **Daytrade**: Continuous fixed-bag fine-tuning on Coinbase live stream
- Smart stagnation detection (variance threshold)
- Automatic service rotation every 6h if training stalls
- Checkpoint pruning (keep last 10 per mode)
- Health checks every 5 minutes

### Cron Schedule
```
*/5   *   *   *   *  # Health check all services
*/30  *   *   *   *  # Check pretrain, rotate if stagnant
0     *   *   *   *  # Ensure daytrade fine-tuning active
0     3   *   *   *  # Full restart of training pipeline
0     4   *   *   *  # Prune old checkpoints
```

### Key Design: Data Separation
- **Pretrain** uses stochastic bags sampled from Binance bulk historical data
- **Daytrade** uses fixed bag.json curated for live Coinbase trading
- No confusion: pretrain never sees live data, daytrade never sees bulk data
- Two distinct checkpoints: `model_weights_pretrained.pt` vs `model_weights_daytrade.pt`

### Acceptance Criteria
- `python3 training_orchestrator.py --daemon` runs continuously
- Services auto-restart within 5 min of failure
- Stagnant training detected and rotated within 30 min
- Defective checkpoints backed up, not loaded
- Daytrade fine-tuning restarts hourly on fresh recent window
- Pretrain continues with new bags after each rotation
