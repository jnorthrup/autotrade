#!/usr/bin/env python3
import argparse

from candle_cache import CandleCache
from config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically bootstrap the canonical candles DB")
    parser.add_argument("--db-path", default=str(Config.DB_PATH))
    parser.add_argument("--granularity", default=Config.DEFAULT_GRANULARITY)
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--future-days", type=int, default=1)
    args = parser.parse_args()

    cache = CandleCache(args.db_path)
    result = cache.bootstrap_database(
        granularity=args.granularity,
        exchange=args.exchange,
        future_days=args.future_days,
    )
    print(
        f"[bootstrap_candles_db] db={result['db_path']} "
        f"pool={result['pool_enabled']} "
        f"exchange={result['exchange']} "
        f"granularity={result['granularity']} "
        f"normalized={result['normalized_timestamps']} "
        f"purged={result['purged_future_rows']}"
    )


if __name__ == "__main__":
    main()
