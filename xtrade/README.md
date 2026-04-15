# xtrade — XChange Kraken Paper Trading Client

A Java-based paper trading client that connects to the Kraken cryptocurrency
exchange via the XChange library. Runs a configurable SMA-based trading
strategy against 5 major pairs (BTC/USD, ETH/USD, XRP/USD, SOL/USD, ADA/USD)
and tracks portfolio balances, unrealized/realized P&L, and fees in a virtual
account.

---

## Prerequisites

- **Java 11** or later (OpenJDK or Oracle JDK)
- **Maven 3.6+** (for building)
- A Kraken account with API credentials (for live mode only; demo mode works
  without any credentials)

---

## Quick Start

### 1. Clone the repository

```bash
cd xtrade
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your Kraken API credentials:

```bash
export KRAKEN_API_KEY="your_api_key"
export KRAKEN_API_SECRET="your_api_secret"
export KRAKEN_MODE="sandbox"   # or "live"
```

Or run in demo mode without any credentials:

```bash
# No environment variables needed — pass --demo flag
```

### 3. Build the project

```bash
mvn clean package
```

This produces a runnable fat JAR at:

```
target/xtrade-1.0-SNAPSHOT.jar
```

### 4. Run the application

**Using the fat JAR:**

```bash
# Demo mode (no API credentials needed):
java -jar target/xtrade-1.0-SNAPSHOT.jar --demo

# Live mode (requires KRAKEN_API_KEY and KRAKEN_API_SECRET):
export KRAKEN_API_KEY="..."
export KRAKEN_API_SECRET="..."
export KRAKEN_MODE="sandbox"
java -jar target/xtrade-1.0-SNAPSHOT.jar
```

**Using Maven exec plugin:**

```bash
mvn exec:java -Dexec.mainClass=com.xtrade.Main -Dexec.args="--demo"
```

---

## Environment Variables

| Variable            | Required | Default    | Description                                       |
|---------------------|----------|------------|---------------------------------------------------|
| `KRAKEN_API_KEY`    | Live     | —          | Your Kraken API key                               |
| `KRAKEN_API_SECRET` | Live     | —          | Your Kraken API secret                            |
| `KRAKEN_MODE`       | No       | `sandbox`  | `sandbox` for paper trading, `live` for real API  |

If `KRAKEN_API_KEY` or `KRAKEN_API_SECRET` is missing, the application
automatically falls back to demo mode with simulated price data.

---

## Build Commands

| Command                     | Description                                |
|-----------------------------|--------------------------------------------|
| `mvn clean compile`         | Compile source code                        |
| `mvn test`                  | Run all unit tests                         |
| `mvn clean package`         | Build fat JAR with all dependencies        |
| `mvn exec:java -Dexec.args="--demo"` | Run in demo mode via Maven       |

---

## Application Configuration

Runtime parameters are configured in `src/main/resources/application.properties`:

| Property                   | Default   | Description                        |
|----------------------------|-----------|------------------------------------|
| `poll-interval-seconds`    | `60`      | Seconds between trading ticks      |
| `initial-virtual-balance`  | `10000.00`| Starting USD balance (paper mode)  |

---

## How to Interpret the Output

### Console Output

Each trading tick prints a summary report table to the console:

```
=====================================================================
            PORTFOLIO SUMMARY REPORT
            Generated: 2026-04-15T03:15:00Z
=====================================================================

  Pair         Qty      Avg Cost      Cur Price   Unrealized P&L       Total P&L
  ----------------------------------------------------------------------------------------

  BTC/USD        0.1000    $50,000.00    $50,000.00         $0.0000         -$13.0000
  ETH/USD             -             -             -                -                -
  XRP/USD             -             -             -                -                -
  SOL/USD        5.0000       $100.00       $100.00         $0.0000          -$1.3000
  ADA/USD             -             -             -                -                -

  Cash Balance:       $4,486.70
  Holdings Value:     $5,500.00
  Total Portfolio:    $9,986.70

  Unrealized P&L:     +$0.0000
  Realized P&L:       +$0.0000
  Total Fees Paid:   $14.3000
  Net Total P&L:      -$14.3000

  Total Trades:       2
=====================================================================
```

### Table Columns

| Column          | Meaning                                                |
|-----------------|--------------------------------------------------------|
| **Pair**        | Trading pair (e.g., BTC/USD)                           |
| **Qty**         | Quantity held; `-` if no position                      |
| **Avg Cost**    | Weighted average cost per unit in USD                  |
| **Cur Price**   | Current market price from last tick                    |
| **Unrealized P&L** | (Current Price - Avg Cost) * Qty for open positions |
| **Total P&L**   | Unrealized P&L + Realized P&L - Fees for the asset    |

### Summary Lines

- **Cash Balance**: Available USD not invested
- **Holdings Value**: Sum of all position values at current prices
- **Total Portfolio**: Cash + Holdings Value
- **Net Total P&L**: Total return including all fees and realized gains/losses
- **Total Trades**: Number of completed buy/sell transactions

### Log Files

Structured log output is written to both the console and a rolling log file:

```
logs/xtrade.log
```

The log file rolls over daily or when it reaches 50 MB, keeping 30 days of
history (up to 1 GB total). Log entries include timestamps, thread names,
log levels, and logger names for easy filtering.

---

## Project Structure

```
xtrade/
├── pom.xml                                  # Maven build with shade plugin
├── .env.example                             # Environment variable template
├── src/
│   ├── main/
│   │   ├── java/com/xtrade/
│   │   │   ├── Main.java                    # Entry point, trading loop
│   │   │   ├── AppConfig.java               # Environment & properties config
│   │   │   ├── ExchangeConfig.java           # Kraken exchange setup
│   │   │   ├── ExchangeService.java          # Market data API with retries
│   │   │   ├── PaperTradingEngine.java       # Virtual trading engine
│   │   │   ├── PortfolioReportPrinter.java   # Formatted console table output
│   │   │   ├── PortfolioSnapshot.java        # Immutable portfolio state
│   │   │   ├── PortfolioPosition.java        # Single position data
│   │   │   ├── PortfolioState.java           # Serializable persistence model
│   │   │   ├── TradingStrategy.java          # Strategy interface
│   │   │   ├── SimpleMovingAverageStrategy.java  # SMA(3,7) crossover
│   │   │   ├── TradingPair.java              # Enum of 5 supported pairs
│   │   │   ├── Signal.java                   # BUY / SELL / HOLD enum
│   │   │   ├── TradeRecord.java              # Immutable trade record
│   │   │   └── LimitOrder.java               # Limit order with status
│   │   └── resources/
│   │       ├── application.properties         # Runtime configuration
│   │       └── logback.xml                    # Logging configuration
│   └── test/java/com/xtrade/
│       ├── MainTest.java
│       ├── PaperTradingEngineTest.java
│       ├── PortfolioReportPrinterTest.java
│       ├── AppConfigTest.java
│       ├── ExchangeConfigTest.java
│       ├── ExchangeServiceTest.java
│       └── MarketDataServiceImplTest.java
├── data/                                     # Portfolio state persistence
│   └── portfolio.json
└── logs/                                     # Log file output
    └── xtrade.log
```

---

## Trading Strategy

The default strategy is a **Simple Moving Average (SMA) crossover** using
short-period (3) and long-period (7) windows:

- **BUY** signal: Short SMA crosses above long SMA
- **SELL** signal: Short SMA crosses below long SMA
- **HOLD**: No crossover detected

Trade size is fixed at $100 USD per signal.

---

## Simulated Fees

The paper trading engine uses a simulated **Kraken taker fee rate of 0.26%**,
applied to both buy and sell orders.

---

## Graceful Shutdown

Press `Ctrl+C` to trigger a graceful shutdown. The engine will:

1. Stop the trading scheduler
2. Persist portfolio state to `data/portfolio.json`
3. Print the final portfolio summary
4. Log the shutdown event

State is automatically restored on the next startup.

---

## License

Internal project — see repository root for license information.
