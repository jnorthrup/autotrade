# Track 4: Stochastic bag sampling with volatility filtering

## Objective
Implement stochastic bag sampling with volatility filtering in graph_showdown.py to replace fixed bag.json usage, enabling better generalization by training on randomly sampled pairs and time windows.

## Tasks

### Task 1: Add utility functions for volatility filtering
- Create `_list_all_binance_pairs()` function to load all pairs from candle_cache
- Create `_compute_volatility_filter()` function to filter pairs by mean |velocity| and drop fiat-fiat edges
- Add FIAT_CURRENCIES constant
- Update imports to include duckdb, datetime, timedelta

### Task 2: Implement stochastic bag sampling
- Add `_BAG_SIZE_SCALE` mapping (4->5, 16->20, 64->40, 256->80)
- Implement `_stochastic_bag_sample()` function that samples pairs based on model size
- Add helper functions: `_build_pair_adjacency()`, `_select_related_pairs()`
- Implement `_make_trial_graph()` to create subgraph from selected pairs and time window

### Task 3: Modify training loop to use stochastic bags
- Replace fixed bag usage with stochastic sampling in `run_autoresearch()`
- Add volatility filtering of pairs before sampling
- Implement stochastic time window sampling (start_bar, end_bar) that scales with training progress
- Add logging for bag composition and window statistics
- Ensure model registration uses edges from trial graph

### Task 4: Verify implementation
- Run smoke test to ensure training still works
- Verify that different pairs are sampled across iterations
- Check that loss is averaged across multiple bag samples per epoch
- Confirm model can generalize to unseen pairs