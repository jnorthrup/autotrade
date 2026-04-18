# Unified workspace reference

This repository now runs as a unified system centered on xtrade.

## Active modules

| Module | Role |
|---|---|
| `xtrade` | Unified trading system: Binance kline ingest, draw-thru cache, paper trading, codec agents, showdown tooling, and feed monitoring |
| `binance` | Supporting Binance client library used by the wider repository |

## Archived components

The old standalone multi-project workspace is archived and excluded from the active Maven reactor. It remains in the repository only as an archive marker and is not the source of truth for runtime behavior.

## Active runtime surfaces

### Kline ingest and cache

- `com.xtrade.kline.DrawThruCachingKlineFeed`
- `com.xtrade.kline.binance.BinanceArchiveFetchService`
- `com.xtrade.kline.binance.BinanceIncrementalFetchService`
- `com.xtrade.kline.binance.BinanceKlineMuxer`
- `com.xtrade.kline.KlineFeedMonitorServer`

### Trading and simulation

- `com.xtrade.Main`
- `com.xtrade.PaperTradingEngine`
- `com.xtrade.kline.PaperTradingEngineKlineAdapter`
- `com.xtrade.kline.CodecPaperTradingAgent`
- `com.xtrade.showdown.ShowdownHarness`

## Operational ownership

Operational documentation lives here:

- README operational guide: `README.md`
- architecture design: `docs/unified-kline-feed-design.md`
- runbook: `docs/kline-feed-operations-runbook.md`

## Monitoring contract

The active feed exports:
- cache hit rate
- feed latency
- producer connectivity
- stale-feed alerts
- `/health` JSON and `/metrics` Prometheus endpoints

## Data layout

```text
xtrade-data/
  cache/klines/<interval>/<base>/<quote>/
  import/klines/<interval>/<base>/<quote>/
```

## Source-of-truth rule

If functionality exists both in an archived legacy location and in xtrade, xtrade is authoritative.
