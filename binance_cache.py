import urllib.request
import zipfile
import io
import pandas as pd
from datetime import datetime
import traceback
import cache
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

def download_month(task):
    binance_pair, tunit, year, month = task
    month_str = f"{month:02d}"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{binance_pair}/{tunit}/{binance_pair}-{tunit}-{year}-{month_str}.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                # prefer first CSV-like entry
                csv_name = None
                for n in z.namelist():
                    if n.lower().endswith(".csv") or n.lower().endswith(".csv.gz"):
                        csv_name = n
                        break
                if csv_name is None:
                    csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5])
                    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
                    return df
    except Exception:
        return None

def fetch_binance_vision(
    db_path: str,
    pair: str = "BTC-USDT",
    granularity: str = "300",
    start_time: datetime = None,
    end_time: datetime = None,
    max_workers: int = 6,
):
    """Fetch monthly Binance Vision klines for `pair` only for months overlapping
    the [start_time, end_time) window. Writes directly to the DuckDB via
    cache.CandleCache.save_candles as results become available.

    This avoids pulling the entire history up front.
    """
    c = cache.CandleCache(db_path=db_path)
    binance_pair = pair.replace("-", "")
    tunit = "5m" if granularity == "300" else "1m"

    # Determine month range
    if start_time is None:
        start = datetime(2017, 1, 1)
    else:
        start = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if end_time is None:
        end = datetime.now()
    else:
        end = end_time

    # build inclusive month list from start..end
    ym = datetime(start.year, start.month, 1)
    months = []
    while ym <= end:
        months.append((ym.year, ym.month))
        if ym.month == 12:
            ym = ym.replace(year=ym.year + 1, month=1)
        else:
            ym = ym.replace(month=ym.month + 1)

    # Determine which months are already present in DB to skip re-download
    missing_months = []
    try:
        import duckdb as _duck

        conn = _duck.connect(db_path, read_only=True)
        for y, m in months:
            start_month = datetime(y, m, 1)
            if m == 12:
                next_month = datetime(y + 1, 1, 1)
            else:
                next_month = datetime(y, m + 1, 1)
            row = conn.execute(
                "SELECT COUNT(*) FROM candles WHERE exchange = ? AND product_id = ? AND granularity = ? AND timestamp >= ? AND timestamp < ?",
                ["binance", pair, granularity, start_month, next_month],
            ).fetchone()
            if not row or row[0] == 0:
                missing_months.append((y, m))
        conn.close()
    except Exception:
        # If any DB error, default to trying all months
        missing_months = months

    if not missing_months:
        print(f"No missing months to fetch for {pair}")
        return

    tasks = [(binance_pair, tunit, y, m) for (y, m) in missing_months]

    print(f"Fetching Binance Vision months for {pair} ({len(tasks)} months) covering {start.date()}..{end.date()}...")

    save_lock = threading.Lock()
    got_any = False
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for df in pool.map(download_month, tasks):
            if df is None:
                # missing month
                sys.stdout.write("x")
                sys.stdout.flush()
                continue
            got_any = True
            df["exchange"] = "binance"
            df["product_id"] = pair
            df["granularity"] = granularity
            df["span"] = 1
            rows = df[["exchange", "product_id", "timestamp", "open", "high", "low", "close", "volume", "granularity", "span"]]
            # Save sequentially to avoid DuckDB file lock contention
            with save_lock:
                try:
                    c.save_candles(rows)
                except Exception:
                    # best-effort: continue
                    pass
            sys.stdout.write(".")
            sys.stdout.flush()

    print()
    if not got_any:
        print("No data found for requested months.")

if __name__ == "__main__":
    for pair in ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT", "BNB-USDT", "LTC-USDT", "DOT-USDT"]:
        fetch_binance_vision("candles.duckdb", pair, "300")
    print("Done!")
