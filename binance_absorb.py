import urllib.request
import zipfile
import io
import logging
import pandas as pd
from datetime import datetime
import traceback
import cache
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

def download_month(task):
    binance_pair, tunit, year, month = task
    month_str = f"{month:02d}"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{binance_pair}/{tunit}/{binance_pair}-{tunit}-{year}-{month_str}.zip"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5])
                    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
                    return df
    except Exception:
        return None

def fetch_binance_vision(db_path: str, pair: str = "BTC-USDT", granularity: str = "300"):
    c = cache.CandleCache(db_path=db_path)
    binance_pair = pair.replace("-", "")
    tunit = "5m" if granularity == "300" else "1m"
    start_year = 2017
    end_year = datetime.now().year

    logger.info("Absorbing Binance Vision for %s %s...", pair, tunit)

    tasks = [(binance_pair, tunit, y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)]

    all_dfs = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for df in pool.map(download_month, tasks):
            if df is not None:
                df["exchange"] = "binance"
                df["product_id"] = pair
                df["granularity"] = granularity
                df["span"] = 1
                all_dfs.append(df[["exchange", "product_id", "timestamp", "open", "high", "low", "close", "volume", "granularity", "span"]])
                sys.stdout.write(".")
                sys.stdout.flush()

    print()
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        logger.info("Saving %d candles to %s...", len(final_df), db_path)
        c.save_candles(final_df)
    else:
        logger.warning("No data found.")

if __name__ == "__main__":
    from logging_config import setup_logging
    setup_logging(level="DEBUG")
    for pair in ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT", "BNB-USDT", "LTC-USDT", "DOT-USDT"]:
        fetch_binance_vision("candles.duckdb", pair, "300")
    logger.info("Done!")
