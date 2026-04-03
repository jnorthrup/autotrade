import json
from datetime import datetime, timedelta

import pandas as pd

import candle_cache as candle_cache_module
from candle_cache import CandleCache
from coin_graph import CoinGraph
from config import Config


class _DummyRESTClient:
    def __init__(self, *args, **kwargs):
        pass


def _align_5m(value: datetime) -> datetime:
    minute = value.minute - (value.minute % 5)
    return value.replace(minute=minute, second=0, microsecond=0)


def _make_rows(exchange: str, product_id: str, start: datetime, offsets: list[int]) -> list[dict]:
    rows = []
    for offset in offsets:
        ts = start + timedelta(minutes=5 * offset)
        price = 100.0 + offset
        rows.append(
            {
                "exchange": exchange,
                "product_id": product_id,
                "timestamp": ts,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
                "volume": 10.0 + offset,
                "granularity": "300",
            }
        )
    return rows


def test_cached_products_in_range_requires_contiguous_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(candle_cache_module, "RESTClient", _DummyRESTClient)
    monkeypatch.setattr(candle_cache_module, "_use_pool", lambda db_path=None: False)
    cache = CandleCache(str(tmp_path / "candles.duckdb"))

    start = _align_5m(datetime.now() - timedelta(hours=1))
    end = start + timedelta(minutes=20)
    rows = []
    rows.extend(_make_rows("coinbase", "BTC-USD", start, [0, 1, 2, 3]))
    rows.extend(_make_rows("coinbase", "ETH-USD", start, [0, 1, 3]))
    cache.save_candles(pd.DataFrame(rows))

    btc_status = cache.range_coverage_status("BTC-USD", start, end, "300", exchange="coinbase")
    eth_status = cache.range_coverage_status("ETH-USD", start, end, "300", exchange="coinbase")

    assert btc_status["covered"] is True
    assert eth_status["covered"] is False
    assert cache.cached_products_in_range(
        ["BTC-USD", "ETH-USD"],
        start,
        end,
        granularity="300",
        exchange="coinbase",
    ) == ["BTC-USD"]


def test_materialized_bag_surface_is_bounded_and_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(candle_cache_module, "RESTClient", _DummyRESTClient)
    monkeypatch.setattr(candle_cache_module, "_use_pool", lambda db_path=None: False)
    cache = CandleCache(str(tmp_path / "candles.duckdb"))

    start = _align_5m(datetime.now() - timedelta(hours=2))
    end = start + timedelta(minutes=20)
    rows = []
    rows.extend(_make_rows("coinbase", "BTC-USD", start, [0, 1, 2, 3, 4]))
    rows.extend(_make_rows("coinbase", "ETH-USD", start, [0, 1, 2, 3]))
    cache.save_candles(pd.DataFrame(rows))

    subscriptions = [
        {"exchange": "coinbase", "product_id": "BTC-USD"},
        {"exchange": "coinbase", "product_id": "ETH-USD"},
    ]
    statuses = cache.verify_bag_contiguous_coverage(subscriptions, start, end, granularity="300")
    assert all(status["covered"] for status in statuses)

    surface_name = cache.materialize_bag_surface(subscriptions, start, end, granularity="300")
    surface_df = cache.read_bag_surface(surface_name)

    assert set(surface_df["product_id"]) == {"BTC-USD", "ETH-USD"}
    assert len(surface_df) == 8
    assert surface_df["timestamp"].max() < pd.Timestamp(end)


def test_repair_http_overlap_forces_overlap_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(candle_cache_module, "RESTClient", _DummyRESTClient)
    monkeypatch.setattr(candle_cache_module, "_use_pool", lambda db_path=None: False)
    cache = CandleCache(str(tmp_path / "candles.duckdb"))

    repair_end = _align_5m(datetime.now())
    calls = []

    def fake_fetch(product_id, exchange, start, end, granularity, force=False, **kwargs):
        calls.append(
            {
                "product_id": product_id,
                "exchange": exchange,
                "start": start,
                "end": end,
                "granularity": granularity,
                "force": force,
            }
        )

    monkeypatch.setattr(cache, "_fetch_and_save", fake_fetch)
    monkeypatch.setattr(
        cache,
        "latest_timestamp",
        lambda product_id, granularity="300", exchange=None: repair_end - timedelta(minutes=10),
    )

    cache.repair_http_overlap(
        ["BTC-USD"],
        granularity="300",
        overlap_candles=6,
        end=repair_end,
        exchange="coinbase",
    )

    assert len(calls) == 1
    # repair_http_overlap now uses backfill_http_exact (direct targeted fetch)
    assert calls[0]["product_id"] == "BTC-USD"
    assert calls[0]["exchange"] == "coinbase"


def test_coin_graph_load_uses_verified_bag_surface(tmp_path, monkeypatch):
    monkeypatch.setattr(candle_cache_module, "RESTClient", _DummyRESTClient)
    monkeypatch.setattr(candle_cache_module, "_use_pool", lambda db_path=None: False)

    db_path = tmp_path / "candles.duckdb"
    cache = CandleCache(str(db_path))
    start = _align_5m(datetime.now() - timedelta(days=1, minutes=5))
    rows = []
    full_day_offsets = list(range(290))
    rows.extend(_make_rows("coinbase", "BTC-USD", start, full_day_offsets))
    rows.extend(_make_rows("coinbase", "ETH-USD", start, full_day_offsets))
    cache.save_candles(pd.DataFrame(rows))

    bag_path = tmp_path / "bag.json"
    bag_path.write_text(
        json.dumps(
            [
                {"exchange": "coinbase", "product_id": "BTC-USD"},
                {"exchange": "coinbase", "product_id": "ETH-USD"},
            ]
        )
    )
    monkeypatch.setattr(Config, "BAG_PATH", bag_path)

    graph = CoinGraph(fee_rate=0.001, min_pair_coverage=1.0)
    n_bars = graph.load(
        db_path=str(db_path),
        granularity="300",
        lookback_days=1,
        use_cached_bag=True,
        skip_fetch=True,
        min_pair_coverage=1.0,
    )

    assert graph.bag_surface_name is not None
    assert n_bars >= 288
    assert sorted(graph.all_pairs) == ["coinbase:BTC-USD", "coinbase:ETH-USD"]
