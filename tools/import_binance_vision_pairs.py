#!/usr/bin/env python3
"""Import symbol list from Binance Public Data (data.binance.vision) or local copies
and populate the DuckDB `pairs` table with BASE-QUOTE product_ids.

This script attempts to parse symbol names from either a local directory of
monthly kline files (downloaded from data.binance.vision) or by attempting to
list the remote index. If the remote index is not browseable, it falls back to
the Binance exchangeInfo API. It uses a suffix-matching heuristic (longest
match first) to split symbols like BTCUSDT -> BTC-USDT.

Usage examples:
  # Parse local files under /data/binance and write to candles.duckdb
  python3 tools/import_binance_vision_pairs.py --local-dir /data/binance --db-path candles.duckdb

  # Try remote listing (will fallback to Binance API if not available)
  python3 tools/import_binance_vision_pairs.py --remote --db-path candles.duckdb

This only writes entries to the `pairs` table; it does not download candlestick
data or change any time ranges.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from typing import List, Optional, Set, Tuple

import duckdb

# Heuristics: known stablecoins, fiat codes, and common quote assets. Order by
# length (longest first) so that USDT is matched before T, etc.
STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "PAX", "TUSD", "USDS", "VAI"}
FIAT_CODES = {
    "USD",
    "EUR",
    "GBP",
    "AUD",
    "BRL",
    "IDR",
    "NGN",
    "KRW",
    "BKRW",
    "RUB",
    "TRY",
    "ZAR",
    "JPY",
    "CNY",
    "UAH",
    "VND",
}

# Site-specific prime fiat aliases that should be treated as fiat for
# classification and routing (PBUSD is a local prime fiat token in some pools).
FIAT_CODES.add("PBUSD")

COMMON_QUOTES = {
    "BTC",
    "ETH",
    "BNB",
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "TRY",
    "EUR",
    "GBP",
    "RUB",
    "JPY",
    "KRW",
}

# Build suffix list, longest-first to prefer multi-letter matches like USDT
SUFFIXES = sorted(list(STABLECOINS | FIAT_CODES | COMMON_QUOTES), key=lambda s: -len(s))

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "LEVERAGED", "ETF", "HEDGE")
LEVERAGED_NUMERIC_SUFFIXES = ("3L", "3S", "4L", "4S", "5L", "5S")


def _is_leveraged_or_etf_symbol(symbol: str, base: str) -> bool:
    """Heuristically reject Binance leveraged token / ETF style symbols.

    We keep short actual coin tickers like JUP; the filter only triggers on
    longer derived-token families such as BTCUP, BTCDOWN, or 3L/3S suffixes.
    """
    symbol_u = (symbol or "").upper()
    base_u = (base or "").upper()
    if not symbol_u or not base_u:
        return False
    if len(base_u) > 4 and any(base_u.endswith(marker) for marker in LEVERAGED_SUFFIXES):
        return True
    if len(base_u) > 4 and any(base_u.endswith(marker) for marker in LEVERAGED_NUMERIC_SUFFIXES):
        return True
    if "LEVERAGED" in symbol_u or "ETF" in symbol_u:
        return True
    return False

EXPECTED_PAIRS_SCHEMA = [
    ("exchange", "VARCHAR"),
    ("product_id", "VARCHAR"),
    ("base", "VARCHAR"),
    ("quote", "VARCHAR"),
    ("quote_type", "VARCHAR"),
    ("leveraged", "BOOLEAN"),
    ("is_etf", "BOOLEAN"),
    ("keep", "BOOLEAN"),
]


def ensure_pairs_schema(conn) -> None:
    """Ensure the DuckDB `pairs` table can hold the full Binance schema.

    Some workspace state has a one-column `pairs(exchange)` table, so we add the
    missing columns in place rather than silently failing inserts.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS pairs (exchange VARCHAR)")
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('pairs')").fetchall()}
    for col_name, col_type in EXPECTED_PAIRS_SCHEMA:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE pairs ADD COLUMN {col_name} {col_type}")


def split_symbol(symbol: str) -> Optional[Tuple[str, str]]:
    """Split an uppercase symbol string into (base, quote) using suffix heuristics.

    Returns None when no reasonable split could be found.
    """
    s = (symbol or "").upper()
    if not s or not s.isalnum():
        return None

    # Try known suffix list (longest matches first)
    for suf in SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            base = s[: len(s) - len(suf)]
            if base and base.isalpha():
                return base, suf

    # Fallback: try plausible suffix lengths (6..2)
    for L in range(6, 1, -1):
        if len(s) > L:
            base, quote = s[: -L], s[-L:]
            if base.isalpha() and quote.isalpha():
                return base, quote

    return None


def collect_local_symbols(local_dir: str) -> List[str]:
    symbols: Set[str] = set()
    for root, dirs, files in os.walk(local_dir):
        # directory names may themselves be symbols
        for d in dirs:
            if d and d.isalnum():
                symbols.add(d.upper())
        for fn in files:
            # files like BTCUSDT-1m-2020-01.zip or BTCUSDT-1h-2020-01.csv.gz
            if not re.search(r"\.(zip|csv|gz)$", fn, re.IGNORECASE):
                continue
            tok = fn.split("-", 1)[0]
            if tok and tok.isalnum():
                symbols.add(tok.upper())
    return sorted(symbols)


def collect_remote_symbols(base_url: str) -> List[str]:
    """Try to build a symbol set from the remote data.binance.vision index.

    Many public buckets are not browseable; in that case fall back to the
    official Binance REST `exchangeInfo` API which lists active symbols.
    """
    symbols: Set[str] = set()
    try:
        with urllib.request.urlopen(base_url, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        # crude extraction of <a href="SYMBOL/"> or occurrences like SYMBOL/
        for m in re.finditer(r'href=["\']?([A-Za-z0-9_\-]+)/', text):
            symbols.add(m.group(1).upper())
        # also look for plain occurrences like >BTCUSDT/
        for m in re.finditer(r">([A-Z0-9]{3,20})/", text):
            symbols.add(m.group(1).upper())
        if symbols:
            return sorted(symbols)
    except Exception:
        # remote listing not available or blocked; we'll fallback below
        pass

    # Fallback to Binance public REST API
    try:
        req = urllib.request.Request(
            "https://api.binance.com/api/v3/exchangeInfo",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        for s in data.get("symbols", []):
            sym = s.get("symbol") or s.get("pair")
            if sym:
                symbols.add(sym.upper())
        return sorted(symbols)
    except Exception:
        return []


def collect_exchangeinfo_pairs() -> List[Tuple[str, str]]:
    """Fetch Binance spot pairs directly from exchangeInfo.

    This is the precise namespace fetch: it uses baseAsset/quoteAsset from the
    API payload instead of trying to recover pairs from a symbol string.
    """
    try:
        req = urllib.request.Request(
            "https://api.binance.com/api/v3/exchangeInfo",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        pairs: List[Tuple[str, str]] = []
        seen = set()
        for s in data.get("symbols", []):
            status = str(s.get("status", "")).upper()
            if status != "TRADING":
                continue
            if not bool(s.get("isSpotTradingAllowed", True)):
                continue
            base = s.get("baseAsset") or s.get("base")
            quote = s.get("quoteAsset") or s.get("quote")
            if not base or not quote:
                continue
            base_u = str(base).upper()
            quote_u = str(quote).upper()
            symbol_u = str(s.get("symbol") or f"{base_u}{quote_u}").upper()
            if _is_leveraged_or_etf_symbol(symbol_u, base_u):
                continue
            pair = (base_u, quote_u)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
        return pairs
    except Exception:
        return []


def write_pairs_to_db(db_path: str, exchange: str, parsed_pairs: List[Tuple[str, str]]) -> int:
    if not parsed_pairs:
        print("No pairs to write")
        return 0

    rows = []
    for base, quote in parsed_pairs:
        pid = f"{base}-{quote}"
        if quote in FIAT_CODES:
            quote_type = "fiat"
        elif quote in STABLECOINS:
            quote_type = "stablecoin"
        else:
            quote_type = "crypto"
        rows.append([exchange, pid, base, quote, quote_type, False, False, True])

    insert_sql = "INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    try:
        conn = duckdb.connect(db_path)
        ensure_pairs_schema(conn)
        try:
            conn.execute("DELETE FROM pairs WHERE exchange = ?", [exchange])
        except Exception:
            pass
        try:
            conn.executemany(insert_sql, rows)
        except Exception:
            for r in rows:
                try:
                    conn.execute(insert_sql, r)
                except Exception:
                    # best-effort: skip problematic rows
                    pass
        conn.close()
        print(f"Imported {len(rows)} pairs for exchange='{exchange}' into {db_path}")
        return len(rows)
    except Exception as direct_exc:
        try:
            import cache

            if cache._use_pool_for_db(db_path):
                pool = cache._pool()
                pool.execute("CREATE TABLE IF NOT EXISTS pairs (exchange VARCHAR)")
                for col_name, col_type in EXPECTED_PAIRS_SCHEMA[1:]:
                    try:
                        pool.execute(f"ALTER TABLE pairs ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
                    except Exception:
                        pass
                try:
                    pool.execute("DELETE FROM pairs WHERE exchange = ?", [exchange])
                except Exception:
                    pass
                inserted = 0
                for r in rows:
                    try:
                        pool.execute(insert_sql, r)
                        inserted += 1
                    except Exception:
                        pass
                print(f"Imported {inserted} pairs for exchange='{exchange}' into {db_path} via pool")
                return inserted
        except Exception as pool_exc:
            print(f"Failed to import pairs via pool for {db_path}: {pool_exc}", file=sys.stderr)
        print(f"Failed to import pairs into DuckDB ({db_path}): {direct_exc}", file=sys.stderr)
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", default=None, help="Local directory containing downloaded Binance Vision files (optional)")
    p.add_argument("--remote", action="store_true", help="Attempt to list remote data.binance.vision index (falls back to Binance API)")
    p.add_argument("--base-url", default="https://data.binance.vision/data/spot/monthly/klines/", help="Base URL for binance vision klines index")
    p.add_argument("--db-path", default="candles.duckdb", help="DuckDB path to write pairs into")
    p.add_argument("--exchange", default="binance", help="Exchange name to write into pairs table")
    args = p.parse_args(argv)

    symbols: List[str] = []
    if args.local_dir:
        if not os.path.isdir(args.local_dir):
            print(f"Local dir not found: {args.local_dir}", file=sys.stderr)
            return 2
        symbols = collect_local_symbols(args.local_dir)
        print(f"Found {len(symbols)} unique symbols in local dir")
    elif args.remote:
        exchange_pairs = collect_exchangeinfo_pairs()
        if exchange_pairs:
            write_pairs_to_db(args.db_path, args.exchange, exchange_pairs)
            return 0
        symbols = collect_remote_symbols(args.base_url)
        print(f"Discovered {len(symbols)} symbols from remote index / exchangeInfo")
    else:
        print("Either --local-dir or --remote must be specified", file=sys.stderr)
        return 3

    parsed = []
    skipped = []
    for sym in symbols:
        res = split_symbol(sym)
        if res:
            parsed.append(res)
        else:
            skipped.append(sym)

    print(f"Parsed {len(parsed)} pairs, skipped {len(skipped)} symbols (ambiguous)")
    if skipped:
        print("Sample skipped symbols:", ", ".join(skipped[:20]))

    write_pairs_to_db(args.db_path, args.exchange, parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
