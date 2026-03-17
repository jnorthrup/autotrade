      ╭─────────────────────────────────────────────────────────────────────────╮
      │ Plan: Force-Directed Coin Graph — Slime Mold / Dijkstra Capital Flow    │
      │              + HRM Hierarchical Reasoning + Autoresearch Loop            │
      │                                                                         │
      │ The Model (in plain physics)                                            │
      │                                                                         │
      │ Capital is fluid. Currencies are nodes. Trading pairs are tubes.        │
      │                                                                         │
      │ - North = high acceleration node (fluid wants to flow here)             │
      │ - South = negative acceleration node (fluid wants to leave)            │
      │ - Tube diameter = conductance (slime mold controlled — grows on profit, │
      │  shrinks on loss)                                                       │
      │ - Dijkstra = finds the cheapest pipe path from current position to      │
      │ north                                                                   │
      │ - Codec agents = pumps — each pushes fluid through a different tube per │
      │  turn                                                                   │
      │ - 1 buy + 1 sell per agent per turn = market maker adjustment, moving   │
      │ the fluid one step                                                      │
      │                                                                         │
      │ The HRM Hierarchy (inspired by Hierarchical Reasoning Model)            │
      │                                                                         │
      │ High-Level Module (slow, abstract planning):                            │
      │   - Computes node heights (north/south potentials) across all nodes     │
      │   - Decides宏观 direction: which currency quadrant is "north"          │
      │   - Outputs: global_direction (bull/bear/neutral per currency)         │
      │   - Update frequency: every 10 bars (slow planning)                    │
      │                                                                         │
      │ Low-Level Module (rapid, detailed execution):                          │
      │   - Computes edge accelerations and velocities                         │
      │   - Executes Dijkstra pathfinding for next trade                       │
      │   - Outputs: specific (base, quote) pair to trade                     │
      │   - Update frequency: every bar (fast execution)                       │
      │                                                                         │
      │ Interdependence:                                                       │
      │   - High-level provides direction context to low-level edge weights    │
      │   - Low-level provides accel feedback to high-level potentials         │
      │   - Two modules communicate via shared graph state                    │
      │                                                                         │
      │ The system self-organizes exactly like an electrical resistor network:  │
      │ - High potential nodes (north) attract current (capital)               │
      │ - Low potential nodes (south) repel it                                 │
      │ - Current follows the path of least resistance (Dijkstra)              │
      │ - Resistances update based on actual flow outcome (slime mold)         │
      │                                                                         │
      │ Node Height (potential) computation                                     │
      │                                                                         │
      │ For each currency node C at bar t:                                      │
      │ height[C] = mean(accel[C→X] for all edges leaving C)                    │
      │            - mean(accel[X→C] for all edges entering C)                  │
      │ Net outflow acceleration = C is gaining vs everything → north.          │
      │ Net inflow acceleration = everything is gaining vs C → south.           │
      │                                                                         │
      │ Edge weight for Dijkstra                                                │
      │                                                                         │
      │ w(A→B) = fee_rate - clip(accel[A→B], -0.01, 0.01) × conductance[A→B]    │
      │ Low weight = wide, profitable tube. Dijkstra finds minimum-cost path.  │
      │ HRM influence: w(A→B) += high_level_direction_penalty if direction conflict│
      │                                                                         │
      │ Slime mold conductance update                                           │
      │                                                                         │
      │ if pnl > 0:  conductance *= (1 + 0.05)   # food — widen the tube      │
      │ if pnl < 0:  conductance *= (1 - 0.03)   # pain — narrow the tube   │
      │ clamp to [0.01, 10.0]                                                   │
      │                                                                         │
      │ Autoresearch Experimentation Loop                                       │
      │                                                                         │
      │ The model trains via autonomous experimentation:                        │
      │ - Experiment budget: 5 minutes wall-clock per iteration                 │
      │ - Modify: train.py hyperparameters, architecture choices                │
      │ - Metric: val_bpb (validation bits per byte — lower is better)          │
      │ - Output: results.tsv with (commit, val_bpb, memory_gb, status, desc)  │
      │ - Loop: experiment → measure → keep if improved / discard if worse     │
      │ - NEVER pause to ask — run indefinitely until manually stopped        │
      │                                                                         │
      │ Files to create                                                         │
      │                                                                         │
      │ coin_graph.py                                                           │
      │                                                                         │
      │ Core model — everything lives here:                                     │
      │ - CoinGraph class                                                       │
      │   - nodes: set of currencies (parsed from candle_cache_5m filenames)    │
      │   - edges: dict of (base, quote) → DataFrame (the 5m candles)           │
      │   - conductance: dict of (base, quote) → float, init 1.0               │
      │   - velocity: dict of edge → float (current 1-bar log return)          │
      │   - accel: dict of edge → float (current acceleration = Δvelocity)    │
      │   - height: dict of node → float (north/south potential)                │
      │ - load(cache_dir): parse all *_5m.pkl files, build graph                │
      │ - update(bar_idx): compute velocity, accel, height for all nodes/edges  │
      │   at bar t                                                              │
      │ - dijkstra(source): standard heapq Dijkstra on edge weights, returns   │
      │ {target: (cost, path)}                                                  │
      │ - best_target(): node with highest height (most north)                  │
      │ - next_hop(source): Dijkstra path[0] — the single next trade           │
      │ - reinforce(base, quote, pnl): slime mold update on that directed edge  │
      │ - node_potentials(): returns sorted list of (height, node) for display  │
      │ - HRM components:                                                       │
      │   - high_level_plan(): returns global_direction per node (slow)         │
      │   - low_level_execute(): returns next trade (fast)                     │
      │   - integrate_hrm_output(): merges high-level context into edge weights│
      │                                                                         │
      │ accel_model.py                                                          │
      │                                                                         │
      │ Tiny online MLP with HRM hierarchy — predicts next accel per edge:     │
      │ - Input: accel values of ALL edges at bar t, last 8 bars → flattened   │
      │ vector                                                                  │
      │ - HRM High-Level: predicts macro direction (currency quadrants)          │
      │   - output: {currency: direction_enum} for each node                  │
      │   - update: every 10 bars                                              │
      │ - HRM Low-Level: predicts per-edge accelerations                        │
      │   - output: predicted accel for each edge at bar t+1                   │
      │   - update: every bar                                                  │
      │ - 3-layer MLP per level: input_dim → 64 → 32 → n_edges                 │
      │ - Online SGD: fit on bar t, predict bar t+1, never look ahead           │
      │ - Used by coin_graph.py to improve edge weights before Dijkstra         │
      │                                                                         │
      │ graph_showdown.py                                                       │
      │                                                                         │
      │ Walk-forward simulation + leaderboard + autoresearch loop:              │
      │ - Load CoinGraph from candle_cache_5m/                                  │
      │ - Walk all common timestamps bar by bar:                                │
      │   - graph.update(t) — recompute accels and heights                      │
      │   - high_level_plan() —宏观 direction every 10 bars                    │
      │   - accel_model.predict() — cross-graph acceleration forecast          │
      │   - integrate_hrm_output() — merge hierarchy into edge weights         │
      │   - dijkstra(holding) — find cheapest path north                         │
      │   - next_hop → 1 trade                                                  │
      │   - Print north/south node potentials every 100 bars                   │
      │   - Execute: record pnl from actual next bar close                     │
      │   - graph.reinforce(edge, pnl)                                          │
      │ - Autoresearch mode (if enabled):                                       │
      │   - experiment_loop(): runs autonomous experiments                      │
      │   - modify: learning_rate, model_width, sequence_length, etc.           │
      │   - metric: val_bpb computed from cumulative pnl                       │
      │   - results.tsv logging after each run                                 │
      │ - Leaderboard tables (persisted to graph_showdown.db):                 │
      │   a. Node table: currency | avg_height | time_at_north% | net_inflow   │
      │   b. Edge table: pair | conductance | n_traversals | cumulative_pnl | │
      │ avg_accel                                                               │
      │   c. Path table: most-used multi-hop routes and their total PnL        │
      │   d. Quote summary: USD/BTC/ETH routing efficiency                     │
      │   e. HRM table: currency | high_level_direction | low_level_accuracy  │
      │ - SQLite schema same pattern as codec_showdown.db                       │
      │                                                                         │
      │ Data flow                                                               │
      │                                                                         │
      │ candle_cache_5m/*.pkl                                                   │
      │       ↓                                                                 │
      │   CoinGraph.load()         — nodes + directed edges from filenames      │
      │       ↓                                                                 │
      │   align timestamps         — inner join all pairs on common 5m bars    │
      │       ↓                                                                 │
      │   walk bar by bar:                                                      │
      │     update(t)              — accel, height for every node              │
      │     high_level_plan()      —宏观 direction (every 10 bars)            │
      │     accel_model.predict()  — cross-graph MLP forecast                  │
      │     integrate_hrm_output() — merge hierarchy into edge weights        │
      │     dijkstra(holding)      — find cheapest path north                   │
      │     next_hop → 1 trade                                           │
      │     record pnl                                                          │
      │     reinforce(edge, pnl)   — slime mold: widen/narrow tubes           │
      │       ↓                                                                 │
      │   graph_showdown.db + printed leaderboard                               │
      │                                                                         │
      │ What the leaderboard shows that codec_showdown couldn't                 │
      │                                                                         │
      │ - Which currencies are persistently north vs south                     │
      │ - Which tubes (pairs) the slime mold widened (positive cumulative PnL)  │
      │ - Which multi-hop routes carried the most capital profitably           │
      │ - Real routing efficiency: does going USD→BTC→SOL beat USD→SOL         │
      │ directly?                                                              │
      │ - HRM performance: high-level direction accuracy vs actual flow        │
      │ - Autoresearch results: best hyperparameters found                     │
      │                                                                         │
      │ Verification                                                            │
      │                                                                         │
      │ python3 fetch_cross_pairs_1m.py   # fills any missing 5m pairs         │
      │ python3 graph_showdown.py          # runs the model, prints leaderboard│
      │ python3 graph_showdown.py --autoresearch  # runs autonomous experiments │
      │                                                                         │
      │ sqlite3 graph_showdown.db "                                             │
      │   SELECT edge, conductance, cumulative_pnl, n_traversals               │
      │   FROM edge_stats ORDER BY conductance DESC LIMIT 20;                   │
      │ "                                                                       │
      │                                                                         │
      │ sqlite3 graph_showdown.db "                                            │
      │   SELECT currency, high_level_direction, low_level_accuracy            │
      │   FROM hrm_stats ORDER BY low_level_accuracy DESC;                      │
      │ "                                                                       │
      │                                                                         │
      │ Files NOT to touch                                                      │
     │                                                                         │


