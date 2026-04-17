# Workspace Reference: mp/*, Binance kline scripts, simawallet, and XChange integration

This document is the shared reference for the workspace. It inventories every mp/* project, catalogs the Binance kline fetch/muxer scripts, and documents the simawallet and XChange integration surfaces that later stories depend on.

## 1) Scope and conventions

Canonical kline schema in this workspace: DataBinanceVision.klines

Canonical path for historical 1m klines:
- mpdata/import/klines/1m/<TC>/<CC>/final-<TC>-<CC>-1m.csv

Canonical ISAM sidecar path used by the kline pipeline:
- same basename as the CSV, with .isam sidecar produced by the cursor/ISAM writers

Legend:
- TC = trade/base asset code, e.g. BTC
- CC = counter/quote asset code, e.g. USDT
- kline = Binance candlestick row
- muxer = a component that combines event streams or row streams into a consolidated candle history

## 2) Complete workspace inventory

| Path | Language | Purpose | Entry points |
|---|---|---|---|
| mp/acapulco | Kotlin | Binance ingest/normalization layer. Downloads historical klines, normalizes Binance CSV rows into the canonical DataBinanceVision schema, consumes live Binance candlestick events, and muxes them into cursor/ISAM views for downstream consumers. | org.bereft.runtime.RealtimeMain.main, org.bereft.node.config.Help.main, org.bereft.node.config.NodeConfig.main, org.bereft.runtime.PrintKeyPair.main, org.bereft.runtime.AsymmetricCryptography.Companion.main, org.bereft.Fail.main |
| mp/control | Kotlin | Offline simulation runtime. Loads imported kline history, runs the trade FSM, and drives the hierarchical reasoning model plus simulated wallet against OHLC snapshots. | org.bereft.Harnass.main |
| mp/databinance | Java + Kotlin | Legacy Binance data-adapter subtree. Bridges Binance candlestick data, cached pair lists, and internal candlestick models. No stable runnable main exists in this snapshot. | None in this snapshot; public APIs are DatabinanceBot and PairsService |
| mp/money | Kotlin + legacy JavaScript | Kraken/Robinhood automation and simwallet persistence. Kotlin ports are the runnable entry points; JS originals remain as reference implementations. | carlos.KrakenSkimmer.main, carlos.Flex.main, carlos.FlexSeth.main; JS originals: carlos/kraken_skimmer.js, carlos/flex.js, carlos/flexseth.js, carlos/simwallet.js, carlos/tradeHistory.js |
| mp/trikeshed-adapter | Kotlin | Cursor, row-vector, ISAM, and columnar support library used by acapulco/control to represent, slice, serialize, and join kline data. | None in this snapshot |
| mp/bin | Bash | Binance archive fetchers, live kline appenders, symbol/pair expansion helpers, and small environment/bootstrap utilities. | fetchklines.sh, dayklines.sh, fetchtrades.sh, meta-klines.sh, allcachedpairs.sh, tweeze.sh, vizGnome.sh, ShardNode.sh, KeyPair.sh, ApiKeyServer.sh, CmcSandboxData.sh |

Notes:
- mp/pom.xml is the parent aggregator for mp/acapulco, mp/control, mp/money, and mp/trikeshed-adapter.
- mp/databinance is present as source, but has no pom.xml in this snapshot.
- mp/bin is a script bundle, not a Maven module.

## 3) Module-by-module reference

### 3.1 mp/acapulco

Language: Kotlin.

Build dependencies of note:
- com.binance.api.client:binance-api-client
- pkg.random:trikeshed-adapter
- kotlinx-coroutines-core-jvm
- kotlinx-datetime-jvm

Core APIs and data models:
- DataBinanceVision: canonical enum of Binance text schemas. It defines aggtrades, klines, and trades. The klines schema is the workspace canonical row layout.
- HistoryService: historical bootstrapper. It discovers imported CSVs, launches fetchklines.sh for missing/stale histories, normalizes opaque CSVs into DataBinanceVision.klines, and writes CSV/ISAM artifacts.
- TradePairEventMuxer: live candle muxer. It buffers incomplete candles in tempKlines, completed candles in latestKlines, converts Binance CandlestickEvent rows to the canonical cursor row shape, and writes ISAM batches when an episode cutoff is reached.
- ITradePairEventMuxer / CandleEventHandler: sink abstractions for candle streams.
- AssetModel: per-asset history store and latest-view cache.
- TradingWallet: Binance-account-backed wallet view used by acapulco-side consumers.
- Streamer / RealtimeStreamer: event bridge and live refresh orchestration.
- KlineViewUtil / MuxIo / KlinePlotThing: display and view-decoration helpers.

How it produces/consumes kline data:
- Consumes Binance websocket CandlestickEvent objects.
- Converts each event into a row vector via CandlestickEvent.row.
- Normalizes historical CSV files via fixOpaqueCsv(..., DataBinanceVision.klines).
- Writes canonical 12-column Binance CSV files.
- Writes ISAM sidecars for completed candle episodes.
- Emits cursor/shared-flow views for downstream simulation and visualization.

Canonical kline row schema (DataBinanceVision.klines):
| Column | Type | Notes |
|---|---|---|
| Open_time | string / unix time | candle open timestamp |
| Open | double | open price |
| High | double | high price |
| Low | double | low price |
| Close | double | close price |
| Volume | double | base asset volume |
| Close_time | string / unix time | candle close timestamp |
| Quote_asset_volume | double | quote volume |
| Number_of_trades | int | trade count |
| Taker_buy_base_asset_volume | double | taker-buy base volume |
| Taker_buy_quote_asset_volume | double | taker-buy quote volume |
| Ignore | string | placeholder/ignored column |

Important entry points:
- org.bereft.runtime.RealtimeMain.main: live Binance runtime entry point.
- org.bereft.node.config.Help.main: prints discovered env/config values.
- org.bereft.node.config.NodeConfig.main: config/debug utility.
- org.bereft.runtime.PrintKeyPair.main: prints generated key material.
- org.bereft.runtime.AsymmetricCryptography.Companion.main: crypto helper demo.
- org.bereft.Fail.main: failure/debug harness.

### 3.2 mp/control

Language: Kotlin.

Build dependencies of note:
- pkg.random:trikeshed-adapter
- pkg.random:acapulco
- ta4j-core (test scope)

Core APIs and data models:
- Simulation: loads imported final CSVs from mpdata/import/klines/1m, discovers tracked assets, opens a shared-flow event stream, and advances simulated time.
- Harnass: top-level RL/HRM driver that constructs the simulation, builds the pancake feature tensor, feeds the hierarchical reasoning model, and pushes resulting trade orders into the trade FSM.
- SimLauncher: opens the simulation stream and drives the coroutine/event loop.
- TradeFSM: candle-driven order execution engine. It keeps a two-snapshot queue per asset, applies fills/cancellations, and executes AssetOutput decisions against OHLC windows.
- SimWallet: control-side simulated wallet. It exposes holdings, a total-value projection, walletFree feature vectors, and order queuing/cancellation semantics.
- TradeOrder: aliased pair of AssetOutput and SimWallet.
- AssetOutput / AssetMutation / AssetKey: decision vector, order-direction enums, and pair identity used by the model.
- pairwise.idiom.hrm.*: the model stack used to derive trade actions.

How it produces/consumes kline data:
- Consumes canonical final-<TC>-<CC>-1m.csv files from mpdata/import/klines/1m.
- Normalizes those CSVs into cursors using HistoryService.fixOpaqueCsv(...).
- Feeds OHLC vectors into TradeFSM.nQ(...) and TradeFSM.transact(...).
- Converts kline windows into model inputs (walletFree + horizon-compressed OHLCV slices).
- Produces simulation-time cursor snapshots for each tick.

Important entry point:
- org.bereft.Harnass.main: offline simulation / RL entry point.

### 3.3 mp/databinance

Language: Java + Kotlin.

Build dependencies of note:
- com.binance.api:binance-api-client
- pkg.random:trikeshed-adapter
- Kotlin stdlib / coroutines / datetime

Core APIs and data models:
- DatabinanceBot: reads an ISAM-cursor-backed history file and converts row vectors back into Binance Candlestick objects for processing or inspection.
- PairsService: pair-to-candlestick helper that returns Map<String, List<CandlestickData>>.
- CandlestickData: local candlestick model with timestamp, open, high, low, close, and volume.
- BinanceMarketDataAdapter: adapter from Binance Candlestick to an internal MarketData-shaped interface.

How it produces/consumes kline data:
- Consumes ISAM-backed candle rows.
- Emits Binance Candlestick objects for downstream inspection/replay.
- Provides a pair-based fetch surface that abstracts over the underlying history store.
- This subtree is legacy/adapter-focused and does not define the canonical kline schema; it consumes the canonical schema produced elsewhere.

Runnable entry points:
- None in this snapshot.

### 3.4 mp/money

Language: Kotlin with legacy JavaScript originals.

Build dependencies of note:
- kotlinx-datetime-jvm
- slf4j-api
- jackson-databind / jackson-module-kotlin / jsr310
- okhttp
- kotlinx-coroutines-core-jvm
- bcprov-jdk15on
- jfreechart / jcommon / xchart

Core APIs and data models:
- KrakenClient: abstraction for live or simulated Kraken access.
- KrakenAPI: public-market-data and private-account helper. Fetches asset/pair metadata, subscribes to public websocket ticker feeds, signs private Balance requests, and exposes live quotes and market orders.
- SimWalletKrakenClient: simwallet wrapper around a live KrakenAPI. Persists cash + holdings to JSON and mirrors market data/order placement while keeping state local.
- TradeHistory / TradeRecord: append-only JSONL trade log.
- SimWallet (Kotlin file): wallet state helpers, persistence seeds, and sim-wallet bootstrap logic.
- Flex: Kotlin port of the token flex bot.
- FlexSeth: placeholder Kotlin main.
- KrakenSkimmer: persistent Kraken portfolio skimmer / cycle loop.
- RobinhoodAPI: Robinhood market/order wrapper for the flex port.

How it produces/consumes kline or price data:
- This subtree is price/portfolio focused rather than kline-focused.
- It consumes live price, holdings, and account state.
- It produces persistent wallet state and trade-history logs.
- It does not define the Binance canonical kline schema.

Important simwallet concepts here:
- Wallet model: cash plus sorted holdings persisted to kraken_simwallet.json or krakenBotState.json.
- Paper trade lifecycle: initialize -> load/bootstrap state -> fetch holdings/quotes -> place simulated buy/sell -> persist updated state.
- Position tracking: holdings are normalized, dust-filtered, and stored per symbol; trades adjust cash and holdings while preserving durable state.

Important entry points:
- carlos.KrakenSkimmer.main
- carlos.Flex.main
- carlos.FlexSeth.main

### 3.5 mp/trikeshed-adapter

Language: Kotlin.

Purpose:
- Structural support layer used by acapulco/control for kline data representation, slicing, joining, remapping, and serialization.

Key APIs and data models:
- cursors.Cursor, cursors.RowVec, cursors.SimpleCursor, cursors.io.ISAMCursor
- cursors.io.writeCSV, cursors.io.writeISAM
- cursors.calendar.UnixTimeRemapper, cursors.context.*, cursors.io.*
- vec.macros.*, vec.util.*

How it produces/consumes kline data:
- Provides the in-memory and on-disk representation used for canonical Binance kline tables.
- Allows the workspace to treat kline history as cursor/ISAM data instead of ad hoc CSVs.

Runnable entry points:
- None in this snapshot.

### 3.6 mp/bin

Language: Bash.

These scripts are the workspace’s practical fetch/mux bootstrap layer.

How they interact with the kline pipeline:
- fetchklines.sh bootstraps historical monthly and daily Binance Vision kline archives into canonical CSV form.
- dayklines.sh tail-appends recent 1m klines to an existing canonical CSV.
- fetchtrades.sh fetches and concatenates Binance trade CSVs.
- meta-klines.sh and tweeze.sh generate large command batches / symbol expansions.
- allcachedpairs.sh discovers imported pair directories.

Script catalog:

#### fetchklines.sh
Location: mp/bin/fetchklines.sh

Purpose:
- Download monthly/daily Binance Vision kline archives, unpack them, deduplicate/sort rows, and write a canonical final CSV.

CLI args and env:
- $1 = TC, default BTC
- $2 = CC, default USDT
- $3 = KLINECACHE override, default $MP_CACHE/klines/$TUNIT/$TC/$CC
- $4 = TARGET override, default $MP_IMPORT/klines/$TUNIT/$TC/$CC
- TIME_UNIT env, default 1m
- MP_CACHE / MP_IMPORT default under ~/mpdata

Output format:
- Writes ${TARGET}/final-${TC}-${CC}-${TUNIT}.csv
- First line is the canonical 12-column Binance kline header
- Rows are newline-delimited CSV entries in Binance kline format

Consumes/produces:
- Consumes Binance Vision monthly/daily .zip archives and their .CHECKSUM files
- Produces canonical 1m kline CSV history used by acapulco/control

#### dayklines.sh
Location: mp/bin/dayklines.sh

Purpose:
- Incrementally append fresh 1m klines from Binance REST onto an existing final CSV.

CLI args and env:
- $1 = TC, default BTC
- $2 = CC, default USDT
- MP_CACHE / MP_IMPORT default under ~/mpdata

Output format:
- Appends rows to ${MP_IMPORT}/klines/1m/${TC}/${CC}/final-${TC}-${CC}-1m.csv
- Uses curl -> sed/tr to transform Binance JSON arrays into CSV rows
- Prints a segments marker and list of temp segment files if the request terminates before the 1000-row limit

Consumes/produces:
- Consumes the last close timestamp from the existing canonical CSV
- Produces appended canonical CSV rows

#### fetchtrades.sh
Location: mp/bin/fetchtrades.sh

Purpose:
- Download monthly/daily Binance trade archives and merge them into a single trade CSV.

CLI args and env:
- $1 = TC, default BTC
- $2 = CC, default USDT
- $3 = BASE override, default ~/mpdata/cache/trades/$TUNIT/$TC/$CC
- $4 = TARGET override, default ~/mpdata/import/trades/$TUNIT/$TC/$CC
- TIME_UNIT env, default 1m

Output format:
- Writes ${TARGET}/${TUNIT}.csv
- First line is a trade header: trade Id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
- Concatenated trade CSV rows from the archive bundle

Consumes/produces:
- Consumes Binance trade archives
- Produces trade-history CSV, not kline CSV

#### meta-klines.sh
Location: mp/bin/meta-klines.sh

Purpose:
- Generate shell command strings that call fetchklines.sh across pair expansions.

CLI args and env:
- Positional args are the traded/base symbols to expand

Output format:
- Prints commands like: for i in <counter list>; do ./fetchklines.sh <BASE> $i ; done

Consumes/produces:
- Consumes symbol lists from tweeze.sh
- Produces shell commands only; it does not fetch data itself

#### tweeze.sh
Location: mp/bin/tweeze.sh

Purpose:
- Static Binance symbol universe helper.
- Provides pair/counter/traded derivations for batch generation.

CLI args and env:
- No required args

Output format:
- pairs(): newline-separated pair list
- counters(): newline-separated counter-currency subset
- traded(): newline-separated traded/base-symbol subset

Consumes/produces:
- Consumes no external data in the script itself
- Produces symbol lists used by meta-klines.sh and other batch scripts

#### allcachedpairs.sh
Location: mp/bin/allcachedpairs.sh

Purpose:
- Discover cached imported pair directories.

CLI args and env:
- No args

Output format:
- Prints directory fragments under ~/mpdata/import that look like pair folders

Consumes/produces:
- Consumes the local import tree only
- Produces a list of cached pairs for orchestration scripts

#### vizGnome.sh
Location: mp/bin/vizGnome.sh

Purpose:
- Empty placeholder in this snapshot.

CLI args and output:
- No operational CLI or output

#### ShardNode.sh
Location: mp/bin/ShardNode.sh

Purpose:
- Launch helper for a shard-node JVM process.

CLI args:
- No script-level args documented in the file; it executes the JVM launcher against a config file

Output format:
- Delegates to the JVM process; no structured output contract in the script itself

#### KeyPair.sh
Location: mp/bin/KeyPair.sh

Purpose:
- Generate and write a keypair into ApiKeys.txt / ShardCfg.txt.

CLI args:
- $1 = mesh id

Output format:
- Appends keypair material and mesh.id mappings to cfg/ApiKeys.txt and cfg/ShardCfg.txt

#### ApiKeyServer.sh
Location: mp/bin/ApiKeyServer.sh

Purpose:
- Launch helper for the API-key node JVM process.

CLI args:
- No script-level args documented in the file; it executes the JVM launcher against cfg/ApiKeys.txt

Output format:
- Delegates to the JVM process; no structured output contract in the script itself

#### CmcSandboxData.sh
Location: mp/bin/CmcSandboxData.sh

Purpose:
- Fetch CoinMarketCap sandbox listing data via curl.

CLI args:
- No positional args

Output format:
- Raw JSON response from the CMC sandbox listings/latest endpoint

## 4) simawallet concepts

This workspace uses two closely related wallet concepts.

### 4.1 Control-side SimWallet (mp/control)

Role:
- SimWallet is the control-side paper wallet used by the HRM / trade-FSM loop.

Wallet model:
- Backed by a SortedMap<String, AssetBalance>.
- Seeds the quote asset with the initial virtual balance, and initializes tracked non-quote assets to zero.
- Exposes holdings, a total-value projection (worth), and a walletFree feature cursor.

Paper trade lifecycle:
1. Simulation bootstraps imported kline history.
2. SimWallet builds the current portfolio state.
3. Harnass derives model features from walletFree + OHLCV history.
4. AssetOutput decisions become TradeOrder values.
5. TradeFSM executes immediate orders or queues limit orders.
6. Order cancellation/expiration is applied on later candles.
7. SimWallet audit records each execution/cancel action.

Position tracking:
- Positions are represented as AssetBalance / AgentAssetBalance rows.
- orderList on a balance holds queued limit orders for that asset.
- applyTrade, plusAssign, cancel, and adjustFree update quantities and audit logs.
- walletFree exposes a normalized per-asset feature vector used by the model.

### 4.2 Money-side simwallet / simulated Kraken client (mp/money)

Role:
- SimWalletKrakenClient wraps a live KrakenAPI and persists local state, enabling paper-trade-like operation around live market data.

Wallet model:
- JSON state contains cash, holdings, and update timestamps.
- Holdings are normalized and dust-filtered before persistence.

Paper trade lifecycle:
1. Load or bootstrap simwallet state.
2. Fetch holdings and market prices from KrakenAPI.
3. Simulated buy/sell mutates local cash and holdings.
4. Persist updated state after each action.

Position tracking:
- Holdings are tracked per normalized symbol.
- Flattened holdings are returned as asset_code / total_quantity rows.
- State survives process restarts via JSON persistence.

## 5) XChange framework integration surface

The xtrade repo uses XChange as its exchange abstraction for Kraken market data.

### 5.1 Exchange adapters

Primary adapter surface:
- ExchangeConfig: reads API credentials and builds a KrakenExchange instance through ExchangeFactory.
- ExchangeService: wraps the XChange Exchange and MarketDataService, exposing retry/rate-limit aware market queries.
- TradingPair: application enum that wraps XChange CurrencyPair values.

What is integrated today:
- KrakenExchange via XChange
- ExchangeFactory / ExchangeSpecification
- CurrencyPair
- Ticker
- OrderBook
- LimitOrder from XChange DTOs for book snapshots
- MarketDataService

### 5.2 Market data interfaces

Current surfaces:
- ExchangeService.getCurrentTicker(CurrencyPair)
- ExchangeService.getOrderBook(CurrencyPair, depth)
- ExchangeService.getAllTickers()
- ExchangeService.getAllOrderBooks(depth)
- MarketDataServiceImpl.fetchAllTickers()
- MarketDataServiceImpl.startPolling() / stopPolling()

Behavior:
- All market-data calls are wrapped with retry logic and rate limiting where appropriate.
- MarketDataServiceImpl is a polling/snapshot wrapper around XChange market data.
- Showdown / realtime data sources can be wired to ExchangeService for live ticker updates.

### 5.3 Order routing

Current status:
- Live order routing is not implemented in xtrade yet.
- ExchangeService is market-data only and explicitly documents that it uses Kraken public endpoints.
- PaperTradingEngine routes market/limit orders internally, but those are simulated orders and do not call XChange order services.

Stable routing-related data models:
- TradeRecord: immutable executed-trade record.
- LimitOrder: paper-order state with OPEN / FILLED / CANCELLED lifecycle.
- PortfolioState and PortfolioSnapshot: persisted and read-only portfolio views.
- PortfolioPosition: per-asset position snapshot.

Practical integration seam for future stories:
- Keep ExchangeService and MarketDataServiceImpl as the XChange boundary.
- Add a dedicated order-service adapter only when live routing is actually needed.
- Do not fork the kline schema for exchange-specific consumers; keep DataBinanceVision.klines canonical and adapt at the boundary.

## 6) Stable cross-module contracts for follow-on stories

1. Keep DataBinanceVision.klines as the canonical Binance 1m candle schema.
2. Keep final-<TC>-<CC>-1m.csv as the canonical historical kline file name.
3. Keep fetchklines.sh as the historical bootstrapper.
4. Keep dayklines.sh as the incremental append path.
5. Treat HistoryService.fixOpaqueCsv(...) as the normalization gate from text CSV to cursor/ISAM forms.
6. Treat SimWallet / TradeFSM / AssetOutput as the control-side simawallet lifecycle.
7. Treat ExchangeService and MarketDataServiceImpl as the XChange market-data boundary.
8. Do not assume live order routing already exists in xtrade; it does not.
9. Prefer adapters over schema forks when connecting new consumers to kline data.
