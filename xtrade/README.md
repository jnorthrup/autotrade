# xtrade

xtrade is the unified trading workspace. Historical Binance klines, live kline muxing, the draw-thru cache, codec agents, showdown tooling, and the paper-trading engine all run from this module.

The legacy standalone workspace has been archived. New ingest, trading, monitoring, and operational changes belong in xtrade.

## What runs here

- Binance archive backfill into canonical CSV
- Incremental day append for recent klines
- Draw-thru in-memory kline cache with producer registration
- Live websocket mux ingestion into the same cache
- Paper-trading engine fed from canonical klines
- Codec showdown and replay tooling
- Feed health, cache metrics, and stale-producer alerting

## Build and test

```bash
cd xtrade
mvn clean test
mvn clean package
```

From the repository root the active Maven modules are `binance` and `xtrade`. The archived legacy workspace is no longer part of the reactor.

## Unified kline architecture

The kline pipeline is built around `com.xtrade.kline.DrawThruCachingKlineFeed`.

1. Producers register ownership of one or more `KlineSeriesId` streams.
2. REST backfill providers satisfy historical cache misses through the same feed.
3. Websocket muxers publish live bars into the same ordered per-series cache.
4. Consumers request history or subscribe for backfill-then-live delivery.
5. Monitoring reads feed metrics directly from the same feed instance and exports `/health` and `/metrics`.

Key classes:

- `DrawThruCachingKlineFeed`: cache, backfill, subscriptions, metrics
- `KlineProducerRegistration`: producer ownership declaration
- `BinanceArchiveFetchService`: historical archive bootstrap
- `BinanceIncrementalFetchService`: recent REST append path
- `BinanceKlineMuxer`: live event normalization and publication
- `PaperTradingEngineKlineAdapter`: bridges kline closes into the paper engine
- `KlineFeedMonitorServer`: HTTP health and Prometheus metrics endpoint

Architecture detail lives in `docs/unified-kline-feed-design.md`.

## Configuration

Runtime configuration comes from `src/main/resources/application.properties` and environment variables.

### Core trading properties

- `poll-interval-seconds`
- `initial-virtual-balance`

### Binance feed properties

- `binance.vision.base-url`
- `binance.api.base-url`
- `binance.kline.start-year`
- `binance.kline.rate-limit-millis`
- `xtrade.kline.cache-root`
- `xtrade.kline.import-root`

### Environment overrides

- `XTRADE_BINANCE_VISION_BASE_URL`
- `XTRADE_BINANCE_API_BASE_URL`
- `XTRADE_BINANCE_START_YEAR`
- `XTRADE_BINANCE_RATE_LIMIT_MILLIS`
- `XTRADE_KLINE_CACHE_ROOT`
- `XTRADE_KLINE_IMPORT_ROOT`

Default storage layout:

```text
~/xtrade-data/
  cache/klines/<interval>/<base>/<quote>/
  import/klines/<interval>/<base>/<quote>/
```

## Binance kline CLI

`com.xtrade.kline.binance.BinanceKlineCli` provides the operational command surface.

Examples:

```bash
# Historical bootstrap
mvn -q -DskipTests exec:java -Dexec.mainClass=com.xtrade.kline.binance.BinanceKlineCli -Dexec.args="fetch BTC USDT --interval 1m"

# Recent append
mvn -q -DskipTests exec:java -Dexec.mainClass=com.xtrade.kline.binance.BinanceKlineCli -Dexec.args="day BTC USDT --interval 1m"

# Quote expansion helper
mvn -q -DskipTests exec:java -Dexec.mainClass=com.xtrade.kline.binance.BinanceKlineCli -Dexec.args="meta BTC ETH"
```

## Feed health and metrics

`DrawThruCachingKlineFeed` exports operational state directly:

- cache requests, hits, misses, and hit rate
- backfill request count and bars loaded
- feed latency: last, average, and max ingest latency
- producer connectivity and last publish age
- stale-producer alerts

Expose the feed over HTTP with `KlineFeedMonitorServer`:

```java
DrawThruCachingKlineFeed feed = new DrawThruCachingKlineFeed(8192);
KlineFeedMonitorServer monitor = KlineFeedMonitorServer.start(
        feed,
        new InetSocketAddress("127.0.0.1", 8088),
        Duration.ofSeconds(90));
```

Endpoints:

- `GET /health` returns JSON health status, cache statistics, and stale-producer alerts
- `GET /metrics` returns Prometheus-compatible metrics for cache hit rate, feed latency, buffered bars, and producer connectivity

Example `/health` status fields:

- `status`: `OK`, `WARN`, or `CRITICAL`
- `metrics.cacheHitRate`
- `metrics.averageFeedLatencyMillis`
- `metrics.producers[*].connected`
- `alerts[*]`

## Operational guide

1. Start or embed the feed monitor server next to the active feed instance.
2. Scrape `/metrics` from Prometheus.
3. Alert when any `xtrade_kline_producer_stale` metric becomes `1`.
4. Alert when `xtrade_kline_cache_hit_rate` drops below the expected warm-cache baseline.
5. Treat `/health` `WARN` or `CRITICAL` as paging signals for feed freshness problems.

Detailed kline feed operations runbook: `docs/kline-feed-operations-runbook.md`.

## Paper trading and showdown

Main entry point:

```bash
mvn exec:java -Dexec.mainClass=com.xtrade.Main -Dexec.args="--demo"
```

Showdown examples:

```bash
java -jar target/xtrade-1.0-SNAPSHOT.jar --showdown --all-codecs --ticks 500 --simulated
java -jar target/xtrade-1.0-SNAPSHOT.jar --showdown --codecs 1,7,14,22 --ticks 100 --simulated
```

## Logs and outputs

- `logs/xtrade.log`
- `data/portfolio.json`
- imported canonical kline CSV under `xtrade.kline.import-root`

## Repository notes

- Active Java development: `xtrade/`
- Supporting Binance client module: `binance/`
- Archived legacy workspace: `mp/` with an archive notice only
