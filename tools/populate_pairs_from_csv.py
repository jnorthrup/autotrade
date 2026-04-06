#!/usr/bin/env python3
import csv, duckdb, os
from collections import OrderedDict

from tools import import_binance_vision_pairs as ibv

STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "PAX", "TUSD", "USDS", "VAI"}
FIAT_CODES = {"USD","EUR","GBP","AUD","BRL","IDR","NGN","KRW","BKRW","RUB","TRY","ZAR","JPY","CNY","UAH","VND"}
COMMON_QUOTES = {"BTC","ETH","BNB","USDT","USDC","BUSD","TUSD","TRY","EUR","GBP","RUB","JPY","KRW"}
SUFFIXES = sorted(list(STABLECOINS | FIAT_CODES | COMMON_QUOTES), key=lambda s: -len(s))


def split_symbol(symbol: str):
    s = (symbol or "").upper()
    if not s or not s.isalnum():
        return None
    for suf in SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            base = s[: len(s) - len(suf)]
            if base and base.isalpha():
                return base, suf
    for L in range(6, 1, -1):
        if len(s) > L:
            base, quote = s[: -L], s[-L:]
            if base.isalpha() and quote.isalpha():
                return base, quote
    return None


csv_path = 'binance_pairs_cleaned.csv'
if not os.path.exists(csv_path):
    print('CSV not found:', csv_path)
    raise SystemExit(2)

symbols = OrderedDict()
with open(csv_path, newline='') as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        sym = (r.get('symbol') or '').strip()
        if sym:
            symbols[sym.upper()] = True

parsed = []
skipped = []
for sym in symbols.keys():
    res = split_symbol(sym)
    if res:
        parsed.append(res)
    else:
        skipped.append(sym)

print('Found', len(symbols), 'symbols in CSV; parsed', len(parsed), 'pairs; skipped', len(skipped))
if skipped:
    print('Sample skipped:', skipped[:20])

rows = []
for base, quote in parsed:
    pid = f"{base}-{quote}"
    if quote in FIAT_CODES:
        quote_type='fiat'
    elif quote in STABLECOINS:
        quote_type='stablecoin'
    else:
        quote_type='crypto'
    rows.append(['binance', pid, base, quote, quote_type, False, False, True])

insert_sql = 'INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
try:
    conn = duckdb.connect('candles.duckdb')
    ibv.ensure_pairs_schema(conn)
    try:
        conn.execute('DELETE FROM pairs WHERE exchange = ?', ['binance'])
    except Exception:
        pass
    try:
        conn.executemany(insert_sql, rows)
    except Exception:
        for r in rows:
            try:
                conn.execute(insert_sql, r)
            except Exception:
                pass

    count = conn.execute("SELECT COUNT(*) FROM pairs WHERE exchange = 'binance'").fetchone()[0]
    print('Imported rows for binance:', count)
    sample = [r[0] for r in conn.execute("SELECT product_id FROM pairs WHERE exchange = 'binance' LIMIT 20").fetchall()]
    print('Sample:', sample)
    conn.close()
except Exception:
    # Could not write directly (likely DuckDB lock). Try talking to pool server
    # over the UNIX socket at /tmp/duckdb_pool.sock (same protocol as PoolClient).
    try:
        import socket, json

        SOCKET = '/tmp/duckdb_pool.sock'

        def _sql_escape(val):
            if val is None:
                return 'NULL'
            if isinstance(val, bool):
                return 'TRUE' if val else 'FALSE'
            if isinstance(val, (int, float)):
                return str(val)
            s = str(val)
            return "'" + s.replace("'", "''") + "'"

        def send_query(sql):
            req = json.dumps({'query': sql}) + '\n'
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(SOCKET)
                sock.sendall(req.encode())
                with sock.makefile('r', encoding='utf-8') as f:
                    line = f.readline()
                    return json.loads(line)
            finally:
                sock.close()

        # Create table and delete existing rows
        send_query("CREATE TABLE IF NOT EXISTS pairs (exchange VARCHAR)")
        send_query("ALTER TABLE pairs ADD COLUMN IF NOT EXISTS product_id VARCHAR")
        send_query("ALTER TABLE pairs ADD COLUMN IF NOT EXISTS base VARCHAR")
        send_query("ALTER TABLE pairs ADD COLUMN IF NOT EXISTS quote VARCHAR")
        send_query("ALTER TABLE pairs ADD COLUMN IF NOT EXISTS quote_type VARCHAR")
        send_query("ALTER TABLE pairs ADD COLUMN IF NOT EXISTS leveraged BOOLEAN")
        send_query("ALTER TABLE pairs ADD COLUMN IF NOT EXISTS is_etf BOOLEAN")
        send_query("ALTER TABLE pairs ADD COLUMN IF NOT EXISTS keep BOOLEAN")
        send_query("DELETE FROM pairs WHERE exchange = 'binance'")

        inserted = 0
        for ex, pid, base, quote, qtype, lev, is_etf, keep in rows:
            sql = (
                'INSERT INTO pairs VALUES ('
                + ", ".join([
                    _sql_escape(ex),
                    _sql_escape(pid),
                    _sql_escape(base),
                    _sql_escape(quote),
                    _sql_escape(qtype),
                    _sql_escape(lev),
                    _sql_escape(is_etf),
                    _sql_escape(keep),
                ])
                + ')'
            )
            try:
                send_query(sql)
                inserted += 1
            except Exception:
                pass
        print('Imported rows for binance via pool socket:', inserted)
    except Exception as e:
        print('Failed to import pairs via pool socket:', e)
