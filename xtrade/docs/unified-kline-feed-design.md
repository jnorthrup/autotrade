# Unified kline model and draw-thru feed SPI

This document defines the single canonical kline contract used by the fetch/mux side and by xtrade consumers. The implementation lives in `com.xtrade.kline` and is intentionally exchange-neutral while preserving exchange metadata required for lossless ingest.

## Goals

1. One bar schema for REST backfills, websocket muxers, file replay, xtrade agents, and the paper trade engine.
2. One draw-thru caching feed that can answer historical requests immediately from memory when possible and backfill on demand when memory is incomplete.
3. One producer SPI for muxers and fetch scripts to register ownership of series and publish canonical bars.
4. One consumer SPI for agents and execution simulators to request history, subscribe to streaming-forward updates, and consume the same bar type everywhere.

## Canonical kline bar schema

Primary types:
- `KlineInterval`: canonical supported intervals from `1s` through `1M`.
- `KlineSeriesId`: `(venue, symbol, interval)` identity for a stream.
- `KlineBar`: immutable canonical bar.

### Fields in `KlineBar`

| Field | Type | Meaning |
|---|---|---|
| `seriesId` | `KlineSeriesId` | Unique stream identity. Includes venue, symbol, interval. |
| `openTimeMillis` | `long` | Inclusive bar-open timestamp aligned to the interval boundary. |
| `closeTimeMillis` | `long` | Exclusive bar-close timestamp. Must equal `openTimeMillis + interval`. |
| `eventTimeMillis` | `long` | Producer event timestamp from websocket/update/replay source. |
| `ingestTimeMillis` | `long` | Local ingest timestamp used for ordering and debugging. |
| `openPrice` | `BigDecimal` | Canonical open. |
| `highPrice` | `BigDecimal` | Canonical high. |
| `lowPrice` | `BigDecimal` | Canonical low. |
| `closePrice` | `BigDecimal` | Canonical close. |
| `baseVolume` | `BigDecimal` | Traded base-asset volume. |
| `quoteVolume` | `BigDecimal` | Traded quote-asset notional. |
| `tradeCount` | `long` | Exchange-reported number of fills/trades in the bar. |
| `takerBuyBaseVolume` | `BigDecimal` | Taker-buy base volume when the venue provides it. |
| `takerBuyQuoteVolume` | `BigDecimal` | Taker-buy quote notional when the venue provides it. |
| `closed` | `boolean` | `true` when the bar is final/closed; `false` for a mutable in-flight mux update. |
| `sequence` | `long` | Producer monotonic sequence per series; later values replace earlier values for the same open time. |
| `producerId` | `String` | Producer that published the bar. |
| `source` | `KlineSource` | `REST_BACKFILL`, `WEBSOCKET_MUX`, `FILE_REPLAY`, `SYNTHETIC_AGGREGATION`, or `MANUAL_IMPORT`. |
| `exchangeMetadata` | `Map<String,String>` | Lossless exchange metadata bag. Examples: exchange symbol, first trade id, last trade id, websocket stream id, timezone, request weight bucket, raw interval token. |

### Required invariants

- `openTimeMillis` is aligned to the `interval` boundary.
- `closeTimeMillis` is exclusive and exactly one interval width after open.
- OHLC prices are positive.
- `highPrice >= openPrice`, `closePrice`, and `lowPrice`.
- `lowPrice <= openPrice` and `closePrice`.
- Volumes and trade count are non-negative.
- `closed=true` is the canonical final bar state used by consumers for trading decisions when they require completed candles only.

### Interval policy

All required intervals are centralized in `KlineInterval`, not scattered across producers or consumers. Producers must map venue-specific wire names to the canonical enum using `KlineInterval.parse(...)`. Consumers can safely key caches, indicator windows, and subscriptions by the enum rather than by raw strings.

## Draw-thru caching feed

Primary types:
- `DrawThruKlineFeed`
- `DrawThruCachingKlineFeed`
- `KlineBatchRequest`
- `KlineSubscriptionRequest`
- `KlineSubscription`

### Semantics

The draw-thru caching feed combines live mux buffering with historical backfill-on-demand.

1. A consumer calls `requestBars(...)` with a `KlineBatchRequest`.
2. If the requested bars already exist in the in-memory series buffer, the feed returns them directly.
3. If the cache is incomplete, the feed discovers the registered producer backfill provider for the requested series and loads the missing range through that producer.
4. Loaded bars are normalized to `KlineBar`, inserted into the same per-series cache, and returned to the caller.
5. Live producer publications flow through the same cache. Subscribers then receive streaming-forward updates from that point onward.

### Buffering and mux behavior

Each `KlineSeriesId` owns one ordered in-memory buffer.
- Bars are keyed by `openTimeMillis`.
- A later publication replaces an earlier publication for the same bar if the later one is closed, has a higher sequence, or has a newer event time.
- The buffer is capped by `maxBarsPerSeries` and evicts oldest bars first.
- Historical backfills and live mux updates share the exact same upsert path, so every consumer sees one canonical series view.

### Historical requests

`KlineBatchRequest` supports two access patterns:
- `between(series, startInclusive, endExclusive)`: explicit historical range.
- `latest(series, limit)`: rolling lookback for agents that need the last N bars.

The `closedBarsOnly` flag lets a consumer decide whether in-flight bars are acceptable. Most indicator agents and paper trading flows should use `closedBarsOnly=true`.

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
- Fetch scripts publish `REST_BACKFILL` bars and expose a `KlineBackfillProvider` implementation for on-demand history.
- Websocket muxers publish `WEBSOCKET_MUX` bars, first with `closed=false` while a candle is still forming, then `closed=true` when the venue closes the candle.
- Replay/import jobs publish `FILE_REPLAY` or `MANUAL_IMPORT` bars.
- All producers must preserve any exchange-specific metadata needed for audits or lossless re-export through the `exchangeMetadata` map.

## Consumer SPI

Primary types:
- `KlineConsumer`
- `DrawThruKlineFeed`
- `PaperTradingEngineKlineAdapter`

### Historical pull

Consumers request historical bars with:
- `requestBars(KlineBatchRequest request)`

This is the pull SPI used by xtrade agents that need warmup windows before they start evaluating indicators.

### Live subscription

Consumers subscribe with:
- `subscribe(KlineSubscriptionRequest request, KlineConsumer consumer)`

`KlineConsumer` receives:
- `onBackfill(KlineSeriesId, List<KlineBar>)`
- `onLiveBar(KlineBar)`
- `onError(KlineSeriesId, Exception)`
- `onClosed(KlineSeriesId)`

This is the push SPI used by:
- xtrade agents that need rolling indicator windows,
- the paper trade engine bridge,
- replay/simulation harnesses.

### Paper trade engine bridge

`PaperTradingEngineKlineAdapter` is the concrete consumer adapter that maps the canonical close price into `PaperTradingEngine.updateMarketPrice(symbol, close)`.

This proves the consumer SPI is sufficient for the execution side:
- historical replay sets the latest market state before trading starts,
- streaming-forward updates keep the engine in sync with live bars.

## Example producer and consumer flow

1. Binance REST fetcher registers series `binance:BTC/USD:1m` with a backfill provider.
2. Binance websocket muxer registers the same series and publishes in-flight plus final bars.
3. An xtrade agent calls `subscribe(backfillThenLive(latest(..., 500)), consumer)`.
4. The feed draws through to the REST producer if the last 500 closed bars are not in memory.
5. The agent receives a single warmup backfill payload.
6. The websocket muxer publishes later bars and the agent receives streaming-forward updates.
7. A paper trading consumer can subscribe to the same series and reuse the exact same canonical bar objects.

## Review checklist

Reviewed:
- canonical schema contains OHLCV, volume, timestamps, intervals, and exchange metadata,
- draw-thru caching feed specifies backfill-on-demand and streaming-forward semantics,
- producer SPI defines registration plus publish behavior,
- consumer SPI defines historical pull plus live subscription behavior,
- implementation and tests live under `com.xtrade.kline` and `src/test/java`.
