# Kline feed operations runbook

## Purpose

This runbook covers the unified xtrade kline feed: ingest, cache health, metrics, and stale-feed alert handling.

## Components

- `DrawThruCachingKlineFeed`
- Binance archive and incremental fetch services
- Binance websocket muxers
- `KlineFeedMonitorServer`

## Startup checklist

1. Confirm the feed has registered all required producers.
2. Confirm cache and import roots point to writable `xtrade-data` paths.
3. Start the monitor server next to the active feed instance.
4. Verify `GET /health` returns `OK` after live traffic begins.
5. Verify Prometheus can scrape `GET /metrics`.

## Health checks

`/health` returns JSON with:
- `status`
- `metrics.cacheRequests`
- `metrics.cacheHitRate`
- `metrics.averageFeedLatencyMillis`
- `metrics.producers[*].connected`
- `metrics.producers[*].lastPublishAgeMillis`
- `alerts`

Operator actions:
- `OK`: normal operation
- `WARN`: partial producer staleness or no registered producers; investigate immediately
- `CRITICAL`: all registered producers are stale; treat as a feed outage

## Metrics to alert on

### Cache hit rate

Metric: `xtrade_kline_cache_hit_rate`

Expected behavior:
- warm services should trend upward after initial backfill
- sustained degradation suggests cache churn, repeated misses, or malformed request patterns

### Feed latency

Metrics:
- `xtrade_kline_feed_latency_millis_avg`
- `xtrade_kline_feed_latency_millis_max`
- `xtrade_kline_feed_latency_millis_last`

Investigate spikes when:
- websocket events arrive late
- downstream publication is blocked
- system load causes ingest delay

### Producer connectivity

Metrics:
- `xtrade_kline_producer_connected{producer_id=...}`
- `xtrade_kline_producer_last_publish_age_millis{producer_id=...}`
- `xtrade_kline_producer_stale{producer_id=...}`

## Recommended alerts

1. Feed staleness alert
   - trigger when `xtrade_kline_producer_stale == 1`
   - severity: page for mandatory market-data producers

2. Health endpoint alert
   - trigger when `/health` is `WARN` or `CRITICAL` for two consecutive scrapes
   - severity: page on `CRITICAL`, ticket on repeated `WARN`

3. Cache efficiency alert
   - trigger when `xtrade_kline_cache_hit_rate` falls below the service baseline for a sustained window
   - severity: investigate during business hours unless paired with latency or staleness symptoms

## Triage flow for stale feed alerts

1. Inspect `/health` and note which producer is stale.
2. Check `xtrade_kline_producer_last_publish_age_millis` for the affected producer.
3. Confirm the websocket muxer or fetch producer is still running.
4. Check recent application logs for transport, auth, or parsing failures.
5. If the producer is dead, restart it and confirm `xtrade_kline_producer_connected` returns to `1`.
6. If the producer is live but lagging, inspect upstream exchange connectivity and local system pressure.

## Recovery verification

Recovery is complete when:
- `/health` returns `OK`
- all mandatory producers report `xtrade_kline_producer_connected == 1`
- all mandatory producers report `xtrade_kline_producer_stale == 0`
- feed latency metrics return to normal operating range
- cache hit rate stabilizes after warmup

## Escalation notes

If stale alerts recur after restart, capture:
- `/health` payload
- `/metrics` scrape
- relevant log excerpts from `logs/xtrade.log`
- producer IDs and affected symbols
