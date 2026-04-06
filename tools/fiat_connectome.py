#!/usr/bin/env python3
"""Port of CoinsAndPairings.fiatConnectome -> simple Python utility.

Reads a cleaned pair list (JSON with `base`/`quote`) and attempts to
find short paths linking each currency to the value asset. Emits a
list of adjacent pairs along found paths that are not already present
in the input set (candidate pairs to add to an asset model's hidden set).

Usage:
  python3 tools/fiat_connectome.py --input binance_pairs_cleaned.json --value USDT

Optional --oracle can be an exchangeInfo JSON (or any JSON with symbols/baseAsset/quoteAsset)
to act as the authoritative connectivity oracle (emulates Binance client data).
"""
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional
import json
import argparse
import sys
import duckdb

from tools import import_binance_vision_pairs as ibv


def load_cleaned(path: Path):
    data = json.loads(path.read_text())
    pairs = []
    for e in data:
        base = e.get("base") or e.get("base_asset")
        quote = e.get("quote") or e.get("quote_asset")
        sym = e.get("symbol")
        if base and quote:
            pairs.append((base.upper(), quote.upper()))
        elif sym and "-" in sym:
            a, b = sym.split("-", 1)
            pairs.append((a.upper(), b.upper()))
        elif sym and sym.isalpha():
            # best-effort: try to split by detecting common quote suffixes
            # fall back: skip
            continue
    return pairs


def load_pairs_from_db(db_path: Path, exchange: Optional[str] = None):
    try:
        conn = duckdb.connect(str(db_path))
        if exchange:
            rows = conn.execute("SELECT base, quote FROM pairs WHERE exchange = ? AND product_id IS NOT NULL", [exchange]).fetchall()
        else:
            rows = conn.execute("SELECT base, quote FROM pairs WHERE product_id IS NOT NULL").fetchall()
        conn.close()
        return [(r[0].upper(), r[1].upper()) for r in rows if r and r[0] and r[1]]
    except Exception:
        return []


def load_exchangeinfo(path: Path):
    data = json.loads(path.read_text())
    pairs = []
    for s in data.get("symbols", []):
        base = s.get("baseAsset") or s.get("base")
        quote = s.get("quoteAsset") or s.get("quote")
        if base and quote:
            pairs.append((base.upper(), quote.upper()))
    return pairs


def build_adj(pairs):
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def shortest_path(adj, start, goal, max_hops=6):
    if start == goal:
        return [start]
    q = deque([[start]])
    seen = {start}
    while q:
        path = q.popleft()
        if len(path) - 1 > max_hops:
            continue
        node = path[-1]
        for nb in adj.get(node, []):
            if nb in seen:
                continue
            seen.add(nb)
            newp = path + [nb]
            if nb == goal:
                return newp
            q.append(newp)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="(optional) path to cleaned pairs JSON; if omitted, pairs will be read from the DuckDB `pairs` table")
    p.add_argument("--exchange", default="binance", help="(optional) exchange to read from DB when --input is omitted (binance|coinbase)")
    p.add_argument("--oracle", default=None, help="exchangeInfo JSON to use as oracle")
    p.add_argument("--value", default="USDT", help="value asset to connect to (e.g. USDT, USD)")
    p.add_argument("--out", default=None, help="(DEPRECATED) path to write connectome additions as JSON; set to None to avoid text files")
    p.add_argument("--db-path", default="candles.duckdb", help="DuckDB path to import connectome additions into (default: candles.duckdb)")
    args = p.parse_args()

    base_pairs = []
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"input not found: {input_path}", file=sys.stderr)
            sys.exit(2)
        base_pairs = load_cleaned(input_path)
    else:
        # Read cleaned pairs from DuckDB `pairs` table
        base_pairs = load_pairs_from_db(Path(args.db_path), exchange=args.exchange)
        if not base_pairs:
            print("No pairs found in DB `pairs` table for exchange", args.exchange, file=sys.stderr)
            sys.exit(3)
    existing_set = set((a, b) for a, b in base_pairs)
    existing_strset = set(f"{a}-{b}" for a, b in base_pairs)

    oracle_pairs = list(base_pairs)
    if args.oracle:
        oracle_path = Path(args.oracle)
        if not oracle_path.exists():
            print(f"oracle not found: {oracle_path}", file=sys.stderr)
            sys.exit(3)
        oracle_pairs = load_exchangeinfo(oracle_path)

    adj = build_adj(oracle_pairs)

    currencies = set()
    for a, b in base_pairs:
        currencies.add(a)
        currencies.add(b)

    value = args.value.upper()
    if value not in adj and value not in currencies:
        print(f"Warning: value asset {value} not found in oracle or input; proceeding anyway")

    added = set()
    for c in sorted(currencies):
        if c == value:
            continue
        path = shortest_path(adj, c, value)
        if not path:
            continue
        # produce adjacent pairs along path
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if (a, b) in existing_set or (b, a) in existing_set:
                continue
            added.add((a, b))

    added_list = [{"base": a, "quote": b, "pair": f"{a}-{b}"} for a, b in sorted(added)]
    print(f"Currencies scanned: {len(currencies)}")
    print(f"Candidate pairs to add: {len(added_list)}")
    if added_list:
        print("Sample additions:")
        for x in added_list[:20]:
            print("  ", x["pair"]) 

    # Import added pairs into DuckDB pairs table for downstream usage (no JSON file)
    db_path = Path(args.db_path)
    try:
        conn = duckdb.connect(str(db_path))
        ibv.ensure_pairs_schema(conn)
        inserted = 0
        for a, b in added:
            pid = f"{a}-{b}"
            # check exists
            try:
                exist = conn.execute("SELECT COUNT(*) FROM pairs WHERE exchange = ? AND product_id = ?", ["connectome", pid]).fetchone()[0]
            except Exception:
                exist = 0
            if not exist:
                try:
                    conn.execute("INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ["connectome", pid, a, b, "", False, False, True])
                    inserted += 1
                except Exception:
                    pass
        conn.close()
        if inserted:
            print(f"Imported {inserted} connectome-added pairs into {db_path}")
    except Exception as e:
        print(f"Failed to import connectome additions into DuckDB ({db_path}): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
