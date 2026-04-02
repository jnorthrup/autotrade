# 002 Exchange-Qualified Candle Store

Status: open
Owner: conductor
Objective: replace pair-only bag subscriptions and pair-only candle identity with an exchange-qualified subscription model rooted in the canonical `candles` table.
Bounded corpus: `candle_cache.py`, `coin_graph.py`, `graph_showdown.py`, `finetune.py`, `training_worker.py`, `tests/test_volatility_filter.py`, `tests/test_training_loop.py`
Runtime route: Codex worker via `spawn_agent`, repo `/Users/jim/work/autotrade`, branch `master`
Stop condition: one slice or first blocker

## Why This Slice Exists

- The canonical DuckDB file was intentionally nuked, so compatibility migration work is no longer the priority.
- `bag.json` still stores bare pairs and cannot tell whether `BTC-USD` means Coinbase or Binance.
- The runtime still treats `product_id` as globally unique across exchanges, which is false.
- `graph_showdown.py` is the active training surface, so subscription and candle truth must be expressed there first.

## Acceptance

- `bag.json` can be parsed as exchange-qualified subscriptions.
- The canonical candle store uses `exchange` in the key and query surfaces.
- The active graph/training loader path consumes bag subscriptions and routes draw-through by exchange.
- Focused tests cover the new candle schema and bag-loading assumptions.
- Canonical candle identity remains the bucket/open timestamp across exchanges; this slice does not introduce a Binance-only close-time key.

## Verification

- `python3 -m py_compile candle_cache.py coin_graph.py graph_showdown.py finetune.py training_worker.py`
- `python3 -m pytest tests/test_volatility_filter.py tests/test_training_loop.py -q`

## Current Verification Evidence

- `python3 -m py_compile candle_cache.py coin_graph.py graph_showdown.py finetune.py training_worker.py` currently passes.
- `python3 -m pytest tests/test_volatility_filter.py tests/test_training_loop.py -q` currently passes (`8 passed`).
- The temp-db pool-routing bug in `graph_showdown.py` is fixed for the targeted helper reads.
- The focused exchange-qualified training-loop tests are now aligned with the active subscription contract.

## Exchange Evidence

- The local `bag.json` file is still a legacy flat string list.
- As of April 1, 2026, all 56 current `bag.json` entries match live Coinbase products.
- Under the Binance importer remap (`USD -> USDT`), 3 entries miss or become invalid: `BAT-ETH`, `CRO-USD`, `USDT-USD`.
- Repo truth for the current local bag is therefore `coinbase`, not `binance`.
