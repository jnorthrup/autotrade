#!/usr/bin/env python3
"""Assemble and clean Binance symbol list from ../mp-superproject.

Reads mp-superproject/mp/bin/symbol.txt and produces cleaned CSV/JSON
classifying quotes and removing leveraged/ETF-like tickers.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json
import csv
import sys
import argparse
import urllib.request
import duckdb


def parse_tweeze(path: Path):
    text = path.read_text()
    lines = text.splitlines()
    syms = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "cat <<" in line:
            # extract delimiter
            after = line.split("<<", 1)[1].strip()
            if not after:
                i += 1
                continue
            delim = after.split()[0].strip().strip("'\"")
            i += 1
            # collect until delim
            while i < len(lines) and lines[i].strip() != delim:
                ln = lines[i].strip()
                if ln and not ln.startswith("#"):
                    syms.append(ln)
                i += 1
            # skip the delim line
            i += 1
            continue
        i += 1
    return syms


SYMBOLS_PATH = Path("/Users/jim/work/mp-superproject/mp/bin/symbol.txt")


def load_symbols(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Symbol list not found: {path}")
    with path.open("r") as f:
        syms = [line.strip() for line in f if line.strip()]
    return syms


def pick_suffix_candidates(symbols, min_count=10, min_len=3, max_len=5):
    cnt = Counter()
    for s in symbols:
        for L in range(min_len, max_len + 1):
            if len(s) > L:
                cnt[s[-L:]] += 1
    # select suffixes that appear reasonably often
    candidates = [s for s, c in cnt.most_common() if c >= min_count]
    # filter out obvious leveraged/synthetic suffixes that would confuse parsing
    bad_substrings = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S")
    filtered = [c for c in candidates if all(bs not in c for bs in bad_substrings) and c.isalpha()]
    candidates = filtered
    # sort by length desc so we match longest suffix first
    candidates = sorted(set(candidates), key=lambda x: (-len(x), -cnt[x], x))
    return candidates


def classify_and_filter(symbols):
    candidates = pick_suffix_candidates(symbols, min_count=8, min_len=3, max_len=6)
    stablecoins = {"USDT", "USDC", "BUSD", "DAI", "PAX", "TUSD", "USDS", "VAI"}
    fiat_codes = {
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

    results = []
    stats = Counter()
    for s in symbols:
        matched = None
        for q in candidates:
            if s.endswith(q) and len(s) > len(q):
                matched = q
                break
        if matched is None:
            # fallback: try lengths 3..6
            for L in range(6, 2, -1):
                if len(s) > L and s[-L:].isalpha():
                    matched = s[-L:]
                    break
        quote = matched or ""
        base = s[: len(s) - len(quote)] if quote else ""

        # detect leveraged / synthetic tokens
        lowered = s.upper()
        leveraged = (
            "UP" in lowered
            or "DOWN" in lowered
            or "BULL" in lowered
            or "BEAR" in lowered
            or "3L" in lowered
            or "3S" in lowered
        )
        is_etf = "ETF" in lowered

        quote_type = "unknown"
        if quote in stablecoins:
            quote_type = "stablecoin"
        elif quote in fiat_codes:
            quote_type = "fiat"
        elif quote:
            quote_type = "crypto"

        keep = not leveraged and not is_etf and quote != ""

        results.append(
            {
                "symbol": s,
                "base": base,
                "quote": quote,
                "quote_type": quote_type,
                "leveraged": leveraged,
                "is_etf": is_etf,
                "keep": keep,
            }
        )
        stats["total"] += 1
        if leveraged:
            stats["leveraged"] += 1
        if is_etf:
            stats["etf"] += 1
        if keep:
            stats["kept"] += 1
        if quote:
            stats[f"quote::{quote}"] += 1

    return results, stats


def write_outputs(results, out_csv: Path, out_json: Path):
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "base", "quote", "quote_type", "leveraged", "is_etf", "keep"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    with out_json.open("w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binance", choices=["binance", "coinbase"], help="Exchange to assemble pairs for")
    parser.add_argument("--fetch-api", action="store_true", help="Fetch current symbols from exchange API instead of using local symbol.txt (Binance) or default API (Coinbase)")
    parser.add_argument("--tweeze", default=None, help="Path to mp/bin/tweeze.sh to extract symbol list from its here-doc (Binance only)")
    parser.add_argument("--exchange-info", default=None, help="Path to a local exchangeInfo JSON file to emulate the exchange client behavior")
    parser.add_argument("--db-path", default="candles.duckdb", help="DuckDB path to import cleaned pairs into (default: candles.duckdb)")
    args = parser.parse_args()
    exchange_info = None
    if args.exchange_info:
        ei_path = Path(args.exchange_info)
        if not ei_path.exists():
            print(f"exchangeInfo file not found: {ei_path}", file=sys.stderr)
            sys.exit(5)
        try:
            exchange_info = json.loads(ei_path.read_text())
        except Exception as e:
            print(f"failed to parse exchangeInfo JSON: {e}", file=sys.stderr)
            sys.exit(6)

    symbols = []
    if args.exchange == "binance":
        if args.tweeze:
            tweeze_path = Path(args.tweeze)
            if not tweeze_path.exists():
                print(f"tweeze file not found: {tweeze_path}", file=sys.stderr)
                sys.exit(4)
            print(f"Parsing symbols from tweeze file: {tweeze_path}")
            symbols = parse_tweeze(tweeze_path)
        elif args.fetch_api:
            print("Fetching symbol list from Binance API (spot exchangeInfo)...")
            try:
                req = urllib.request.Request("https://api.binance.com/api/v3/exchangeInfo", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    data = json.load(resp)
                symbols = [sobj["symbol"] for sobj in data.get("symbols", [])]
            except Exception as e:
                print("Failed to fetch from Binance API:", e, file=sys.stderr)
                sys.exit(3)
        else:
            try:
                symbols = load_symbols(SYMBOLS_PATH)
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                sys.exit(2)
    else:
        # Coinbase: fetch product list from Coinbase Exchange (recommended)
        if args.exchange_info:
            ei_path = Path(args.exchange_info)
            if not ei_path.exists():
                print(f"exchangeInfo file not found: {ei_path}", file=sys.stderr)
                sys.exit(5)
            try:
                data = json.loads(ei_path.read_text())
                # Expect list of symbols or products
                symbols = [s.get("id") or s.get("symbol") for s in data.get("symbols", data) if (s.get("id") or s.get("symbol"))]
            except Exception as e:
                print(f"failed to parse exchangeInfo JSON: {e}", file=sys.stderr)
                sys.exit(6)
        elif args.fetch_api:
            print("Fetching product list from Coinbase Exchange API...")
            try:
                req = urllib.request.Request("https://api.exchange.coinbase.com/products", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    data = json.load(resp)
                symbols = [p.get("id") for p in data if p.get("id")]
            except Exception as e:
                print("Failed to fetch from Coinbase API:", e, file=sys.stderr)
                sys.exit(3)
        else:
            print("No local symbol source for Coinbase specified; use --fetch-api or --exchange-info", file=sys.stderr)
            sys.exit(2)

    # Build results using a uniform schema: symbol/base/quote/etc.
    results = []
    stats = Counter()
    if args.exchange == "binance":
        results, stats = classify_and_filter(symbols)
    else:
        # Coinbase products: parse id -> base-quote
        for pid in symbols:
            if not pid or "-" not in pid:
                continue
            base, quote = pid.split("-", 1)
            lowered = pid.upper()
            leveraged = False
            is_etf = False
            quote_type = "fiat" if quote in {"USD", "EUR", "GBP"} else "crypto"
            keep = True
            results.append({
                "symbol": pid,
                "base": base,
                "quote": quote,
                "quote_type": quote_type,
                "leveraged": leveraged,
                "is_etf": is_etf,
                "keep": keep,
            })
            stats["total"] += 1

    # Persist cleaned pairs to DuckDB only (no text/json files)

    # summary
    print(f"Total symbols: {stats['total']}")
    print(f"Leveraged/synthetic: {stats.get('leveraged',0)}")
    print(f"ETFs: {stats.get('etf',0)}")
    print(f"Kept (non-leveraged, has quote): {stats.get('kept',0)}")

    # unique quotes and top quotes
    quotes = {k.split("::", 1)[1]: v for k, v in stats.items() if k.startswith("quote::")}
    top_quotes = sorted(quotes.items(), key=lambda x: -x[1])[:20]
    print("Top quotes:")
    for q, c in top_quotes:
        print(f"  {q}: {c}")

    # Note: outputs written into DuckDB below; no CSV/JSON files emitted

    # Import cleaned pairs into DuckDB so downstream tooling can query them.
    db_path = Path(args.db_path)
    try:
        conn = duckdb.connect(str(db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pairs (exchange VARCHAR, product_id VARCHAR, base VARCHAR, quote VARCHAR, quote_type VARCHAR, leveraged BOOLEAN, is_etf BOOLEAN, keep BOOLEAN)"
        )
        insert_sql = "INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        rows = []
        for r in results:
            rows.append([
                args.exchange,
                r.get("symbol"),
                r.get("base"),
                r.get("quote"),
                r.get("quote_type"),
                bool(r.get("leveraged")),
                bool(r.get("is_etf")),
                bool(r.get("keep")),
            ])
        # Replace any existing rows for this exchange so imports are idempotent
        try:
            conn.execute("DELETE FROM pairs WHERE exchange = ?", [args.exchange])
        except Exception:
            pass
        # Use executemany via duckdb Python - fallback to individual inserts
        try:
            conn.executemany(insert_sql, rows)
        except Exception:
            for row in rows:
                try:
                    conn.execute(insert_sql, row)
                except Exception:
                    # best-effort: skip problematic rows
                    pass
        conn.close()
        print(f"Imported {len(rows)} cleaned pairs into {db_path}")
    except Exception as e:
        # If direct DuckDB write fails (e.g., file lock by pool), try using the pool client
        tried_pool = False
        try:
            import cache
            # Try to ensure pool is running for this DB (will start pool if needed)
            try:
                cache.ensure_pool_running(str(db_path))
            except Exception:
                # ignore; we'll try to use pool client if it's available
                pass
            if cache._use_pool_for_db(str(db_path)):
                tried_pool = True
                pool = cache._pool()
                pool.execute(
                    "CREATE TABLE IF NOT EXISTS pairs (exchange VARCHAR, product_id VARCHAR, base VARCHAR, quote VARCHAR, quote_type VARCHAR, leveraged BOOLEAN, is_etf BOOLEAN, keep BOOLEAN)"
                )
                try:
                    pool.execute("DELETE FROM pairs WHERE exchange = ?", [args.exchange])
                except Exception:
                    pass
                for row in rows:
                    try:
                        pool.execute(insert_sql, row)
                    except Exception:
                        pass
                print(f"Imported {len(rows)} cleaned pairs into {db_path} via pool")
        except Exception:
            pass
        if not tried_pool:
            print(f"Failed to import cleaned pairs into DuckDB ({db_path}): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
