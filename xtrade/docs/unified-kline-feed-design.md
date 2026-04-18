# Unified kline model and draw-thru feed SPI

This document defines the canonical kline contract used across xtrade producers, consumers, paper trading, and operational monitoring.

## Goals

1. One bar schema for REST backfills, websocket muxers, file replay, xtrade agents, and the paper trade engine.
2. One draw-thru caching feed that can answer historical requests from memory when possible and backfill on demand when memory is incomplete.
3. One producer SPI for fetchers and muxers to register ownership of series and publish canonical bars.
4. One consumer SPI for agents and execution simulators to request history, subscribe to streaming-forward updates, and consume the same bar type everywhere.
5. One operational monitoring surface for health, cache hit rates, feed latency, producer connectivity, and stale-feed alerting.

## Canonical kline bar schema

Primary types:
- `KlineInterval`: canonical supported intervals from `1s` through `1M`
- `KlineSeriesId`: `(venue, symbol, interval)` identity for a stream
- `KlineBar`: immutable canonical bar

### Fields in `KlineBar`

| Field | Type | Meaning |
|---|---|---|
| `seriesId` | `KlineSeriesId` | Unique stream identity. Includes venue, symbol, interval. |
| `openTimeMillis` | `long` | Inclusive bar-open timestamp aligned to the interval boundary. |
| `closeTimeMillis` | `long` | Exclusive bar-close timestamp. Must equal `openTimeMillis + interval`. |
| `eventTimeMillis` | `long` | Producer event timestamp from websocket, replay, or import source. |
| `ingestTimeMillis` | `long` | Local ingest timestamp used for ordering, metrics, and debugging. |
| `openPrice` | `BigDecimal` | Canonical open. |
| `highPrice` | `BigDecimal` | Canonical high. |
| `lowPrice` | `BigDecimal` | Canonical low. |
| `closePrice` | `BigDecimal` | Canonical close. |
| `baseVolume` | `BigDecimal` | Traded base-asset volume. |
| `quoteVolume` | `BigDecimal` | Traded quote-asset notional. |
| `tradeCount` | `long` | Exchange-reported number of fills/trades in the bar. |
| `takerBuyBaseVolume` | `BigDecimal` | Taker-buy base volume when available. |
| `takerBuyQuoteVolume` | `BigDecimal` | Taker-buy quote notional when available. |
| `closed` | `boolean` | `true` when the bar is final; `false` for in-flight mux updates. |
| `sequence` | `long` | Producer monotonic sequence per series. |
| `producerId` | `String` | Producer that published the bar. |
| `source` | `KlineSource` | `REST_BACKFILL`, `WEBSOCKET_MUX`, `FILE_REPLAY`, `SYNTHETIC_AGGREGATION`, or `MANUAL_IMPORT`. |
| `exchangeMetadata` | `Map<String,String>` | Lossless exchange metadata bag. |

### Required invariants

- `openTimeMillis` is aligned to the `interval` boundary.
- `closeTimeMillis` is exclusive and exactly one interval width after open.
- OHLC prices are positive.
- `highPrice >= openPrice`, `closePrice`, and `lowPrice`.
- `lowPrice <= openPrice` and `closePrice`.
- Volumes and trade count are non-negative.
- `closed=true` is the canonical final bar state for closed-candle consumers.

## Draw-thru caching feed

Primary types:
- `DrawThruKlineFeed`
- `DrawThruCachingKlineFeed`
- `KlineBatchRequest`
- `KlineSubscriptionRequest`
- `KlineSubscription`

### Semantics

1. A consumer calls `requestBars(...)` with a `KlineBatchRequest`.
2. If the requested bars already exist in the in-memory series buffer, the feed returns them directly.
3. If the cache is incomplete, the feed discovers the registered producer backfill provider and loads the missing range.
4. Loaded bars are normalized to `KlineBar`, inserted into the same per-series cache, and returned to the caller.
5. Live producer publications flow through the same cache. Subscribers then receive streaming-forward updates from that point onward.

### Buffering and mux behavior

Each `KlineSeriesId` owns one ordered in-memory buffer.
- Bars are keyed by `openTimeMillis`.
- A later publication replaces an earlier publication for the same bar if the later one is closed, has a higher sequence, or has a newer event time.
- The buffer is capped by `maxBarsPerSeries` and evicts oldest bars first.
- Historical backfills and live mux updates share the exact same upsert path.

### Historical requests

`KlineBatchRequest` supports two access patterns:
- `between(series, startInclusive, endExclusive)`
- `latest(series, limit)`

The `closedBarsOnly` flag lets a consumer decide whether in-flight bars are acceptable. Most indicator agents and paper-trading flows should use `closedBarsOnly=true`.

### Streaming-forward subscriptions

`KlineSubscriptionRequest` supports:
- `liveOnly(series)`
- `backfillThenLive(request)`

`subscribe(...)` behavior:
1. Register the subscription for the series.
2. If the request includes replay, fetch the historical set via the same `requestBars(...)` path.
3. Deliver `onBackfill(seriesId, bars)` once.
4. Forward every subsequent matching publication through `onLiveBar(bar)`.
5. On unsubscribe, remove the listener and emit `onClosed(seriesId)`.

That is the required backfill-on-demand plus streaming-forward contract.

## Producer SPI

Primary types:
- `KlineProducerRegistration`
- `KlineProducerHandle`
- `KlineBackfillProvider`

### Registration

A producer registers with:
- `producerId`
- `producerName`
- `publishedSeries`
- optional `backfillProvider`

Call:
- `registerProducer(KlineProducerRegistration registration)`

Registration is strict. A producer may only `publish(...)` or `publishAll(...)` for series it declared in `publishedSeries`.

### Publishing

A muxer or fetcher obtains a `KlineProducerHandle` after registration.

Calls:
- `publish(KlineBar bar)`
- `publishAll(Collection<KlineBar> bars)`

Producer expectations:
- Fetchers publish `REST_BACKFILL` bars and expose `KlineBackfillProvider` for on-demand history.
- Websocket muxers publish `WEBSOCKET_MUX` bars, first with `closed=false` while a candle is still forming, then `closed=true` when the venue closes the candle.
- Replay or import jobs publish `FILE_REPLAY` or `MANUAL_IMPORT` bars.
- All producers preserve exchange metadata needed for audits or re-export.

## Consumer SPI

Primary types:
- `KlineConsumer`
- `DrawThruKlineFeed`
- `PaperTradingEngineKlineAdapter`

### Historical pull

Consumers request historical bars with:
- `requestBars(KlineBatchRequest request)`

### Live subscription

Consumers subscribe with:
- `subscribe(KlineSubscriptionRequest request, KlineConsumer consumer)`

`KlineConsumer` receives:
- `onBackfill(KlineSeriesId, List<KlineBar>)`
- `onLiveBar(KlineBar)`
- `onError(KlineSeriesId, Exception)`
- `onClosed(KlineSeriesId)`

### Paper trade engine bridge

`PaperTradingEngineKlineAdapter` maps the canonical close price into `PaperTradingEngine.updateMarketPrice(symbol, close)`.

## Operational monitoring design

Primary types:
- `KlineFeedMetricsSnapshot`
- `KlineFeedHealthReport`
- `KlineFeedProducerStatus`
- `KlineFeedMonitorServer`

### Exported metrics

The feed exports:
- cache requests, hits, misses, and cache hit rate
- backfill request count and bars loaded
- feed latency measured from `eventTimeMillis` to `ingestTimeMillis`
- buffered bar count and active subscription count
- producer connectivity, last publish age, and published bar totals

### Health semantics

`healthReport(Duration staleAfter)` returns:
- `OK` when all producers are publishing within the freshness threshold
- `WARN` when at least one producer is stale or when no producers are registered
- `CRITICAL` when every registered producer is stale

Alerts are included as human-readable strings so they can be logged or surfaced by runbooks.

### Prometheus metrics endpoint

`KlineFeedMonitorServer` exposes:
- `/health` returning JSON health plus cache statistics
- `/metrics` returning Prometheus text format

Representative metrics:
- `xtrade_kline_cache_hit_rate`
- `xtrade_kline_feed_latency_millis_avg`
- `xtrade_kline_feed_latency_millis_max`
- `xtrade_kline_producer_connected{producer_id=...}`
- `xtrade_kline_producer_stale{producer_id=...}`

### Alerting on feed staleness

Recommended alerts:
1. Page when `xtrade_kline_producer_stale == 1` for any mandatory producer.
2. Page when `/health` returns `WARN` or `CRITICAL` for more than one scrape interval.
3. Investigate cache regressions when `xtrade_kline_cache_hit_rate` drops materially below the expected warm-cache baseline.

## Example producer and consumer flow

1. Binance REST fetcher registers series `binance:BTC/USDT:1m` with a backfill provider.
2. Binance websocket muxer registers the same series and publishes in-flight plus final bars.
3. An xtrade agent calls `subscribe(backfillThenLive(latest(..., 500)), consumer)`.
4. The feed draws through to the REST producer if the last 500 closed bars are not in memory.
5. The agent receives a single warmup backfill payload.
6. The websocket muxer publishes later bars and the agent receives streaming-forward updates.
7. The monitor endpoint exports cache hit rate, feed latency, producer connectivity, and staleness status for the same feed instance.

## Review checklist

Reviewed:
- canonical schema contains OHLCV, volume, timestamps, intervals, and exchange metadata
- draw-thru caching feed specifies backfill-on-demand and streaming-forward semantics
- producer SPI defines registration plus publish behavior
- consumer SPI defines historical pull plus live subscription behavior
- `papertradingengineklineadapter` proves execution-side consumption
- monitoring exports health, cache hit rate, feed latency, producer connectivity, and stale-feed alerts
- implementation and tests live under `com.xtrade.kline` and `src/test/java`
