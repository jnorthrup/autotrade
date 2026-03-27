import pytest
from graph_showdown import _list_all_binance_pairs, _compute_volatility_filter, FIAT_CURRENCIES
import duckdb
import os
from datetime import datetime, timedelta

def test_fiat_currencies_constant():
    """Test that FIAT_CURRENCIES is a set with expected currencies."""
    assert isinstance(FIAT_CURRENCIES, set)
    assert "USD" in FIAT_CURRENCIES
    assert "USDT" in FIAT_CURRENCIES
    assert "BTC" not in FIAT_CURRENCIES  # BTC is not fiat

def test_list_all_binance_pairs_empty_db(tmp_path):
    """Test _list_all_binance_pairs returns empty list when DB has no candles."""
    db_path = tmp_path / "test.duckdb"
    # Create empty DB
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE candles (product_id VARCHAR, timestamp TIMESTAMP, open DOUBLE, close DOUBLE, granularity VARCHAR)")
    conn.close()
    
    pairs = _list_all_binance_pairs(str(db_path))
    assert pairs == []

def test_list_all_binance_pairs_with_data(tmp_path):
    """Test _list_all_binance_pairs returns correct pairs."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE candles (product_id VARCHAR, timestamp TIMESTAMP, open DOUBLE, close DOUBLE, granularity VARCHAR)")
    conn.execute("INSERT INTO candles VALUES ('BTC-USD', '2024-01-01', 100, 101, '300')")
    conn.execute("INSERT INTO candles VALUES ('ETH-USD', '2024-01-01', 200, 202, '300')")
    conn.execute("INSERT INTO candles VALUES ('ETH-BTC', '2024-01-01', 0.05, 0.051, '300')")
    conn.execute("INSERT INTO candles VALUES ('USD-EUR', '2024-01-01', 1, 1.01, '300')")  # fiat-fiat
    conn.execute("INSERT INTO candles VALUES ('INVALID', '2024-01-01', 1, 1, '300')")  # invalid format
    conn.close()
    
    pairs = _list_all_binance_pairs(str(db_path))
    # Should return sorted unique valid pairs, excluding fiat-fiat? Actually _list_all_binance_pairs does not filter fiat-fiat, that's done in _compute_volatility_filter
    # So we expect all valid format pairs: BTC-USD, ETH-USD, ETH-BTC, USD-EUR
    expected = sorted(["BTC-USD", "ETH-USD", "ETH-BTC", "USD-EUR"])
    assert pairs == expected

def test_compute_volatility_filter_filters_fiat_fiat(tmp_path):
    """Test that _compute_volatility_filter removes fiat-fiat pairs."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE candles (product_id VARCHAR, timestamp TIMESTAMP, open DOUBLE, close DOUBLE, granularity VARCHAR)")
    # Insert data with high velocity for all pairs
    now = datetime.now()
    start = now - timedelta(days=1)
    for pid in ["BTC-USD", "ETH-USD", "USD-EUR"]:
        conn.execute(f"INSERT INTO candles VALUES ('{pid}', '{start}', 100, 101, '300')")
        conn.execute(f"INSERT INTO candles VALUES ('{pid}', '{now}', 100, 101, '300')")
    conn.close()
    
    all_pairs = ["BTC-USD", "ETH-USD", "USD-EUR"]
    filtered = _compute_volatility_filter(str(db_path), all_pairs, lookback_days=1, min_velocity=0.0)
    # USD-EUR should be filtered out because both are fiat
    assert "USD-EUR" not in filtered
    assert "BTC-USD" in filtered
    assert "ETH-USD" in filtered

def test_compute_volatility_filter_by_velocity(tmp_path):
    """Test that _compute_volatility_filter filters by mean |velocity|."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE candles (product_id VARCHAR, timestamp TIMESTAMP, open DOUBLE, close DOUBLE, granularity VARCHAR)")
    now = datetime.now()
    start = now - timedelta(days=1)
    # High velocity pair
    conn.execute("INSERT INTO candles VALUES ('BTC-USD', ?, 100, 110, '300')", [start])
    conn.execute("INSERT INTO candles VALUES ('BTC-USD', ?, 100, 110, '300')", [now])
    # Low velocity pair
    conn.execute("INSERT INTO candles VALUES ('ETH-USD', ?, 100, 100.5, '300')", [start])
    conn.execute("INSERT INTO candles VALUES ('ETH-USD', ?, 100, 100.5, '300')", [now])
    conn.close()
    
    all_pairs = ["BTC-USD", "ETH-USD"]
    # With min_velocity=0.05, only BTC-USD should pass (approx 0.095 > 0.05)
    filtered = _compute_volatility_filter(str(db_path), all_pairs, lookback_days=1, min_velocity=0.05)
    assert "BTC-USD" in filtered
    assert "ETH-USD" not in filtered

if __name__ == "__main__":
    pytest.main([__file__, "-v"])