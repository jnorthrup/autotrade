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
│ - Tube diameter = conductance (slime mold controlled — grows on profit,│
│   shrinks on loss)                                                       │
│ - Dijkstra = finds the cheapest pipe path from current position to      │
│   north                                                                   │
│ - Codec agents = pumps — each pushes fluid through a different tube    │
│   per turn                                                               │
│ - 1 buy + 1 sell per agent per turn = market maker adjustment, moving │
│   the fluid one step                                                    │
│                                                                         │
│ The HRM Hierarchy (inspired by Hierarchical Reasoning Model)            │
│                                                                         │
│ High-Level Module (slow, abstract planning):                            │
│   - Computes node heights (north/south potentials) across all nodes    │
│   - Decides macro direction: which currency quadrant is "north"         │
│   - Outputs: global_direction (bull/bear/neutral per currency)         │
│   - Update frequency: every 10 bars (slow planning)                    │
│                                                                         │
│ Low-Level Module (rapid, detailed execution):                           │
│   - Computes edge accelerations and velocities                          │
│   - Executes Dijkstra pathfinding for next trade                        │
│   - Outputs: specific (base, quote) pair to trade                      │
│   - Update frequency: every bar (fast execution)                        │
│                                                                         │
│ Interdependence:                                                         │
│   - High-level provides direction context to low-level edge weights     │
│   - Low-level provides accel feedback to high-level potentials         │
│   - Two modules communicate via shared graph state                     │
│                                                                         │
│ The system self-organizes exactly like an electrical resistor network:  │
│   - High potential nodes (north) attract current (capital)             │
│   - Low potential nodes (south) repel it                                │
│   - Current follows the path of least resistance (Dijkstra)             │
│   - Resistances update based on actual flow outcome (slime mold)        │
│                                                                         │
│ Node Height (potential) computation                                     │
│                                                                         │
│ For each currency node C at bar t:                                      │
│   height[C] = mean(accel[C→X] for all edges leaving C)                 │
│ Net outflow acceleration = C is gaining vs everything → north.         │
│                                                                         │
│ Edge weight for Dijkstra                                                │
│                                                                         │
│   w(A→B) = fee_rate - accel[A→B] × conductance[A→B] + hrm_penalty       │
│ Low weight = wide, profitable tube. Dijkstra finds minimum-cost path.  │
│ HRM influence: penalty added when direction conflicts with high-level  │
│                                                                         │
│ Slime mold conductance update                                           │
│                                                                         │
│   if pnl > 0:  conductance *= 1.02   # food — widen the tube           │
│   if pnl < 0:  conductance *= 0.90   # pain — narrow the tube          │
│   clamp to [0.01, 10.0]                                                   │
│                                                                         │
│ Autoresearch Experimentation Loop                                       │
│                                                                         │
│ The model trains via autonomous experimentation:                        │
│   - Experiment budget: configurable (default 5 minutes per iteration)   │
│   - Modify: learning_rate, model_width, y_depth, x_pixels, curvature    │
│   - Metric: val_bpb (validation bits per byte — lower is better)       │
│     Computed as: -pnl / n_trades (negative PnL per trade)              │
│   - Output: DuckDB experiments table with (timestamp, val_bpb, params)│
│   - Loop: experiment → measure → keep if improved / discard if worse   │
│   - NEVER pause to ask — run indefinitely until manually stopped       │
│                                                                         │
│ Files (Core Implementation)                                            │
│                                                                         │
│ config.py                                                                │
│                                                                         │
│ Configuration management:                                               │
│   - Loads environment variables from .env file                         │
│   - COINBASE_API_KEY, COINBASE_API_SECRET                             │
│   - DB_PATH (DuckDB database path)                                     │
│   - DEFAULT_GRANULARITY (default bar granularity = 300 = 5min)         │
│   - Validation warnings for missing API keys                           │
│                                                                         │
│ candle_cache.py                                                          │
│                                                                         │
│ DuckDB-based candle caching with rate limiting:                          │
│   - CandleCache class:                                                  │
│     - get_candles(product_id, start, end, granularity): retrieve       │
│     - prefetch_all(pairs, start, end): concurrent prefetch + 429 handle │
│     - ws_snapshot(pairs, granularity): WebSocket snapshot for live    │
│     - _TokenBucket: global rate limiter (7 req/s safe limit)            │
│     - _init_db(): creates candles table with indices                    │
│     - _fetch_and_save(): handles chunked API requests with retry       │
│   - Database schema:                                                    │
│     - candles table: product_id, timestamp, OHLCV, granularity        │
│     - PRIMARY KEY: (product_id, timestamp, granularity)                │
│     - Indices: (product_id, granularity), timestamp                   │
│                                                                         │
│ coin_graph.py                                                            │
│                                                                         │
│ Core model — everything lives here:                                     │
│   - CoinGraph class                                                     │
│     - nodes: set of currencies (discovered from Coinbase live)        │
│     - edges: dict of (base, quote) → DataFrame (the 5m candles)         │
│     - edge_state: dict of (base, quote) → EdgeState                    │
│       - conductance, velocity, accel, cumulative_pnl, n_traversals    │
│       - short_term_memory, long_term_memory, streak, temperature       │
│       - volatility_window, wins, losses                                 │
│     - node_state: dict of currency → NodeState                         │
│       - height, net_inflow, time_at_north, total_bars                  │
│     - common_timestamps: aligned 5m bar timestamps across all pairs    │
│   - load(db_path, granularity, min_partners): discover + load data     │
│   - hydrate_increment(days): pull more data from cache/API             │
│   - update(bar_idx): compute velocity, accel, height for all nodes/edges│
│   - _compute_heights(edge_accels): calculate node potentials           │
│   - dijkstra(source): standard heapq Dijkstra on edge weights          │
│   - best_target(): node with highest height (most north)               │
│   - next_hop(holding, target): Dijkstra path[0] — next single trade    │
│   - reinforce(base, quote, pnl): slime mold update on directed edge    │
│   - node_potentials(): returns sorted list of (height, node)           │
│   - get_edge_stats(): returns edge statistics for leaderboard          │
│   - get_node_stats(): returns node statistics for leaderboard          │
│   - HRM components:                                                     │
│     - high_level_plan(): returns global_direction per node (slow)      │
│     - low_level_execute(holding, predicted_accels): returns next trade │
│     - integrate_hrm_output(hrm_direction): merges high-level into weights│
│   - edge_weight(base, quote): compute Dijkstra weight with HRM penalty │
│                                                                         │
│ accel_model.py                                                           │
│                                                                         │
│ Graph Attention Network (GAT) with HRM hierarchy and fisheye:           │
│   - EdgeEncoder: encode per-edge fisheye features to 128-dim embedding  │
│   - GraphAttentionLayer: single round of message passing with attention │
│   - GraphAttentionNetwork: full GAT for velocity prediction             │
│     Input: per-edge fisheye features (y_depth candles → x_pixels)      │
│     Metadata: curvature, y_depth, x_pixels, fee_rate                   │
│     Output: predicted velocity per edge                                │
│   - Fisheye compression:                                                │
│     - fisheye_boundaries(y_depth, x_pixels, curvature): non-linear    │
│     - fisheye_sample(candles, boundaries): sample using bucket boundaries│
│     - Learned per-edge: curvature, y_depth, x_pixels (PyTorch params)  │
│   - HRM High-Level: predicts macro direction per currency node          │
│     Output: {currency: direction_enum} (north/south/neutral)          │
│     Update: every 10 bars (from node potentials)                       │
│   - HRM Low-Level: predicts per-edge velocities using GAT              │
│     Output: predicted velocity for each edge at bar t+1               │
│     Update: every bar (online SGD)                                     │
│   - AccelModel class:                                                   │
│     - register_edges(edges): build graph structure, init fisheye params│
│     - _build_input_tensor(graph): build per-edge feature matrix        │
│     - high_level_plan(graph): infer direction from node potentials     │
│     - predict(graph): predict per-edge velocity using GAT              │
│     - update(graph, actual_accels, bar_idx): update GAT with actuals   │
│     - update_prices(graph, bar_idx): update close price buffer         │
│     - get_hrm_stats(): return HRM statistics for leaderboard          │
│                                                                         │
│ graph_showdown.py                                                        │
│                                                                         │
│ Walk-forward simulation + leaderboard + autoresearch loop:             │
│   - WSCandles class: aggregates live ticks into 5-min OHLCV bars       │
│     Subscribes to Coinbase WebSocket for all pairs                    │
│     Injects closed bars into graph.edges and common_timestamps        │
│   - run_simulation(): walk-forward bar-by-bar execution               │
│     REVEAL bar t actual data                                           │
│     COLLECT PnL for trade decided at bar t-1                          │
│     TRAIN model on bar t (no future leakage)                          │
│     PREDICT bar t+1 using model trained through bar t                  │
│     DECIDE trade for bar t+1 based on predictions                      │
│     Supports PortfolioManager modes: single_asset, fractional, multi  │
│   - run_autoresearch(): autonomous experimentation loop                │
│     Random hyperparameter search: lr, width, y_depth, x_pixels, curve │
│     Run simulation, compute val_bpb = -pnl / n_trades                  │
│     Persist to DuckDB experiments table                                │
│     Track best parameters and infinite loop until Ctrl+C              │
│   - persist_leaderboard_stats(): save run stats to DuckDB              │
│   - _init_leaderboard_tables(): create stats tables                   │
│   - Leaderboard output (printed to console):                           │
│     Final Results: capital, PnL%, trades, avg gain per trade          │
│     Node Potentials (Top 10): sorted by height                         │
│     Edge Stats (Top 10 by conductance): pair, conductance, cum_pnl    │
│     Multi-Hop Path Stats: most-used routes and total PnL              │
│     Quote Routing Efficiency: USD/BTC/ETH routing stats              │
│                                                                         │
│ portfolio_manager.py                                                     │
│                                                                         │
│ Advanced portfolio management with multiple strategies:                │
│   - PortfolioManager class:                                             │
│     - Modes: single_asset, fractional (Kelly), multi_asset (rank flows│
│     - decide(): decide trade based on mode, confidence, Kelly fraction │
│     - rank_all_flows(): rank all flows (long winners AND short losers) │
│     - execute(): execute trade decision, update positions               │
│     - record_pnl(): record PnL for Kelly computation                   │
│     - get_holding(): get current holding currency                      │
│     - get_portfolio_value_usd(): compute total value via shortest path  │
│     - get_stats(): return portfolio statistics                          │
│   - TradeDecision dataclass: base, quote, fraction, kelly, confidence  │
│   - Position dataclass: currency, size, entry_value                   │
│                                                                         │
│ Supporting Scripts                                                       │
│                                                                         │
│ fetch_cross_pairs_1m.py                                                  │
│                                                                         │
│ Cross-pair candle ingestion for crypto-quoted pairs:                   │
│   - CrossPairIngestor class                                             │
│   - get_cross_products(): discover pairs where quote is BTC/ETH/SOL/etc│
│   - ingest_product(): fetch and save candles to DuckDB                 │
│   - Focuses on lateral paths for Dijkstra routing                       │
│   - Usage: python3 fetch_cross_pairs_1m.py --days 30 --top 20          │
│                                                                         │
│ train_model.py                                                           │
│                                                                         │
│ Batch training with Graph Attention Network:                           │
│   - build_dataset(): pre-compute fisheye features and velocities        │
│   - split(): 70/15/15 train/val/test temporal split                    │
│   - train_epoch(): GAT training with MSE loss                          │
│   - evaluate(): compute MSE and directional accuracy                   │
│   - simulate_test(): walk-forward test set simulation                   │
│   - Saves best model to velocity_model.pt                               │
│                                                                         │
│ autoresearch/ (Standalone Autoresearch System)                          │
│                                                                         │
│ A separate autonomous experiment framework with different approach:   │
│   - auto.py: Main loop with iterative hyperparameter optimization       │
│   - train.py: Training script with configurable depth, aspect_ratio    │
│   - prepare.py: Data preparation utilities                             │
│   - Runs experiments with depth, aspect_ratio, matrix_lr, embedding_lr │
│   - Tracks val_bpb and memory usage in results.tsv                      │
│                                                                         │
│ Other utilities:                                                         │
│   - fetch_1y_1m.py: fetch 1 year of 1-minute candles                   │
│   - ingest.py: general candle ingestion utility                        │
│   - tail_results.py: view experiment results from DuckDB                │
│                                                                         │
│ Data flow                                                                │
│                                                                         │
│ Coinbase WebSocket / candles.duckdb                                    │
│       ↓                                                                 │
│   CoinGraph.load()         — discover graph + load data                │
│       ↓                                                                 │
│   align timestamps         — inner join all pairs on common 5m bars    │
│       ↓                                                                 │
│   walk bar by bar:                                                      │
│     update(t)              — accel, height for every node              │
│     high_level_plan()      — macro direction per node                  │
│     update_prices()        — feed close prices to fisheye buffer       │
│     update()               — train GAT on actual velocities           │
│     predict()              — GAT predicts next velocities              │
│     integrate_hrm_output() — merge HRM direction into edge weights     │
│     dijkstra(holding)      — find cheapest path north                  │
│     next_hop → 1 trade                                              │
│     record pnl                                                         │
│     reinforce(edge, pnl)   — slime mold: widen/narrow tubes           │
│       ↓                                                                 │
│   printed leaderboard + DuckDB experiments table                        │
│                                                                         │
│ What the implementation shows that codec_showdown couldn't             │
│                                                                         │
│ - Graph-based capital flow: currencies as nodes, pairs as edges         │
│ - Which currencies are persistently north vs south (node potentials)   │
│ - Which tubes (pairs) the slime mold widened (positive cumulative PnL)  │
│ - Which multi-hop routes carried the most capital profitably           │
│ - Real routing efficiency: does going USD→BTC→SOL beat USD→SOL         │
│   directly?                                                             │
│ - HRM performance: high-level direction vs actual flow                 │
│ - GAT with fisheye compression: learned per-edge time series encoding │
│ - Portfolio management: single asset, Kelly fraction, multi-asset       │
│ - Live trading: WebSocket support for real-time bar injection          │
│ - Autoresearch results: best hyperparameters found via experiments     │
│                                                                         │
│ Verification                                                             │
│                                                                         │
│ # Fetch historical data                                                 │
│ python3 fetch_cross_pairs_1m.py   # fills any missing 5m pairs         │
│                                                                         │
│ # Run simulation                                                        │
│ python3 graph_showdown.py --pm-mode single_asset --initial-capital 100 │
│ python3 graph_showdown.py --pm-mode fractional --initial-capital 1000  │
│ python3 graph_showdown.py --pm-mode multi_asset --initial-capital 1000│
│                                                                         │
│ # Run autoresearch (infinite loop)                                     │
│ python3 graph_showdown.py --autoresearch                               │
│                                                                         │
│ # Live trading (after history exhausted)                               │
│ python3 graph_showdown.py --live --pm-mode single_asset               │
│                                                                         │
│ # Query experiments table                                              │
│ python3 -c "import duckdb; conn = duckdb.connect('candles.duckdb');    │
│     print(conn.execute('SELECT * FROM experiments ORDER BY val_bpb    │
│         DESC LIMIT 10').df())"                                         │
│                                                                         │
│ # Standalone autoresearch (separate system)                            │
│ cd autoresearch && python3 auto.py                                     │
│                                                                         │
│ Database Schema (DuckDB: candles.duckdb)                                │
│                                                                         │
│ CREATE TABLE IF NOT EXISTS experiments (                                │
│   timestamp TIMESTAMP DEFAULT now(),                                    │
│   val_bpb DOUBLE,                                                       │
│   params VARCHAR                                                        │
│ );                                                                      │
│                                                                         │
│ CREATE TABLE IF NOT EXISTS node_stats (                                │
│   run_id VARCHAR, timestamp TIMESTAMP, currency VARCHAR,               │
│   avg_height DOUBLE, time_at_north_pct DOUBLE, net_inflow DOUBLE,      │
│   total_bars INTEGER                                                    │
│ );                                                                      │
│                                                                         │
│ CREATE TABLE IF NOT EXISTS edge_stats (                                │
│   run_id VARCHAR, timestamp TIMESTAMP, edge VARCHAR,                   │
│   conductance DOUBLE, cumulative_pnl DOUBLE, n_traversals INTEGER,    │
│   avg_accel DOUBLE, last_pnl DOUBLE                                    │
│ );                                                                      │
│                                                                         │
│ CREATE TABLE IF NOT EXISTS path_stats (                                │
│   run_id VARCHAR, timestamp TIMESTAMP, path VARCHAR,                   │
│   n_uses INTEGER, total_pnl DOUBLE                                     │
│ );                                                                      │
│                                                                         │
│ CREATE TABLE IF NOT EXISTS quote_routing_stats (                       │
│   run_id VARCHAR, timestamp TIMESTAMP, quote_currency VARCHAR,         │
│   n_trades INTEGER, total_pnl DOUBLE                                   │
│ );                                                                      │
│                                                                         │
│ CREATE TABLE IF NOT EXISTS hrm_stats (                                 │
│   run_id VARCHAR, timestamp TIMESTAMP, currency VARCHAR,               │
│   high_level_direction VARCHAR, low_level_accuracy DOUBLE              │
│ );                                                                      │
│                                                                         │
│ Note: Leaderboard stats are printed to console.                        │
│       Persistent database tables for node/edge/path/stats are created │
│       by _init_leaderboard_tables() and populated by persist_leaderboard_stats()│
│                                                                         │
│ Files NOT to touch                                                       │
│                                                                         │
│ The following files are supporting utilities or data files and should  │
│ not be modified without understanding their purpose:                   │
│                                                                         │
│   - candle_cache_5m: Cached candle data file (binary pickle format)    │
│   - candles.duckdb: DuckDB database containing all historical candles  │
│   - velocity_model.pt: Trained GAT model weights (generated by training│
│   - autoresearch.log: Autoresearch experiment log                      │
│   - .env: Local environment configuration (API keys, secrets)          │
│   - .venv/: Python virtual environment directory                       │
│   - __pycache__/: Python bytecode cache                                │
│   - .git/: Git version control metadata                                │
│                                                                         │
│ Environment Setup                                                        │
│                                                                         │
│ 1. Copy .env.example to .env and add your Coinbase API credentials    │
│ 2. Install dependencies: pip install -r requirements.txt              │
│ 3. Run initial data fetch: python3 fetch_cross_pairs_1m.py --days 30   │
│ 4. Run simulation: python3 graph_showdown.py --pm-mode single_asset    │
│                                                                         │
│ Requirements                                                             │
│                                                                         │
│   - Python 3.9+                                                         │
│   - coinbase-restsdk (Coinbase Advanced SDK)                           │
│   - coinbase-websocket-sdk (WebSocket client)                          │
│   - torch (PyTorch for GAT model)                                      │
│   - duckdb (database)                                                  │
│   - pandas, numpy (data manipulation)                                  │
│   - python-dotenv (environment configuration)                          │
╰─────────────────────────────────────────────────────────────────────────╯