# Showdown – Multi-Agent Codec Trading Simulation

A command-line harness for running head-to-head comparisons of trading codec
agents against simulated or historical price data.

## Quick Start

```bash
# Demo mode – 200 ticks of simulated random-walk prices with 5 agents:
python -m showdown.cli --agents 01,14,15,13,03 --ticks 200 --balance 10000 --demo

# Historical replay from CSV:
python -m showdown.cli --agents 01,14,15,13,03 --data prices.csv --balance 50000

# Verbose mode with custom pairs and seed:
python -m showdown.cli --agents 01,02,03 --pairs BTC/USDT,ETH/USDT --ticks 500 --demo --seed 99 --verbose

# Export results to JSON:
python -m showdown.cli --agents 01,14,15,13,03 --ticks 200 --balance 10000 --demo --export results.json
```

## Command-Line Arguments

| Argument    | Description                                                            | Default                       |
|-------------|------------------------------------------------------------------------|-------------------------------|
| `--agents`  | Comma-separated codec IDs (01-24)                                      | `01,14,15,13,03,02`           |
| `--pairs`   | Comma-separated trading pairs                                          | `BTC/USDT`                    |
| `--ticks`   | Number of simulation cycles                                            | `200`                         |
| `--balance` | Starting cash per agent                                                | `10000`                       |
| `--data`    | Price history CSV file for replay (columns: timestamp,pair,price,volume) | (none)                        |
| `--demo`    | Use simulated random-walk prices (identical to xtrade's demo mode)     | `false`                       |
| `--seed`    | Random seed for demo mode                                              | `42`                          |
| `--verbose` | Print progress every 10 ticks                                          | `false`                       |
| `--export`  | Export results to a JSON file                                          | (none)                        |

## Available Codec Agents

| ID | Name                     | Strategy                          |
|----|--------------------------|-----------------------------------|
| 01 | volatility_breakout      | Volatility breakout detection     |
| 02 | momentum_trend           | Momentum-based trend following    |
| 03 | mean_reversion           | Mean reversion trading            |
| 04 | trend_following          | Classic trend following           |
| 05 | pairs_trading            | Statistical pairs trading         |
| 06 | grid_trading             | Grid trading                      |
| 07 | volume_profile           | Volume profile analysis           |
| 08 | order_flow               | Order flow imbalance              |
| 09 | correlation_trading      | Cross-asset correlation           |
| 10 | liquidity_making         | Market making / liquidity provision|
| 11 | sector_rotation          | Sector-based rotation             |
| 12 | composite_alpha          | Multi-signal composite alpha      |
| 13 | rsi_reversal             | RSI-based reversal strategy       |
| 14 | bollinger_bands          | Bollinger Bands mean reversion    |
| 15 | macd_crossover           | MACD crossover signals            |
| 16 | stochastic_kd            | Stochastic %K/%D oscillator       |
| 17 | adx_trend_strength       | ADX trend strength filter         |
| 18 | vwap_mean_reversion      | VWAP mean reversion               |
| 19 | kalman_filter_trend      | Kalman filter trend estimation    |
| 20 | hurst_regime             | Hurst exponent regime detection   |
| 21 | random_forest_classifier | ML random forest classifier       |
| 22 | xgboost_signal           | XGBoost signal model              |
| 23 | transformer_attention    | Transformer attention model        |
| 24 | zscore_stat_arb          | Z-score statistical arbitrage     |

## Expected Output

Running:

```bash
python -m showdown.cli --agents 01,14,15,13,03 --ticks 200 --balance 10000 --demo
```

Produces output with these sections:

### 1. Banner
Shows configuration: agents, pairs, ticks, balance, data source.

### 2. Leaderboard
A table sorted by total return with columns: Rank, Agent, Return%, Sharpe, HitRate, Trades, MaxDD.

```
================================================================================
LEADERBOARD (sorted by total return)
================================================================================
Rank  Agent                       Return%   Sharpe  HitRate  Trades    MaxDD
--------------------------------------------------------------------------------
   1  bollinger_bands              22.77%  678.055 100.00%      77   4.54%
   2  mean_reversion               20.57%  704.760 100.00%      50   4.29%
   3  macd_crossover                0.12%   29.133  18.89%     195  15.68%
   4  volatility_breakout           0.00%      N/A   0.00%       0   0.00%
   4  rsi_reversal                  0.00%      N/A   0.00%       0   0.00%
================================================================================
```

### 3. Equity Curve Comparison
ASCII sparklines showing each agent's equity trajectory with min/max/final values.

### 4. Per-Agent Breakdown
Detailed stats for each agent: initial cash, final value, P&L, return %, Sharpe,
hit rate, max drawdown, trade counts, ticks processed.

### 5. Overall Summary
Aggregate stats: number of agents, total trades, average/median return,
best and worst performers.

## Data Sources

### Demo Mode (`--demo`)
Generates synthetic random-walk prices using geometric Brownian motion.
Parameters: drift=0.0001, volatility=0.02 per tick. This matches the xtrade
Java app's demo mode output.

### Historical Replay (`--data prices.csv`)
Replays prices from a CSV file with columns: `timestamp`, `pair`, `price`, `volume`.
Rows with the same timestamp are grouped into a single tick, allowing multi-pair
data in a single file.

### CSV Format Example
```csv
timestamp,pair,price,volume
1700000000.0,BTC/USDT,42350.50,1250.00
1700000001.0,BTC/USDT,42375.25,980.50
```

## Programmatic Usage

```python
from showdown.runner import ShowdownRunner
from showdown.report import build_agent_reports, print_leaderboard

runner = ShowdownRunner(
    codec_ids=[1, 14, 15, 13, 3],
    data_source="simulated",
    initial_cash=10000.0,
    pairs=["BTC/USDT"],
    num_ticks=200,
)
runner.run()
reports = build_agent_reports(runner)
print_leaderboard(reports)
```

## Running Tests

```bash
python -m pytest tests/test_cli.py -v
```
