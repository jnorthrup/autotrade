# DuckDB Pool Integration

## Status

Active local component in this repo.

The DuckDB pool is owned by autotrade under `./duckdb-pool`. It is not a
cross-repo dependency.

## Purpose

Autotrade uses a singleton DuckDB pool process to avoid mixed-config
connection conflicts and to give the Python training/runtime processes one
shared query surface.

## Layout

```text
autotrade/
├── duckdb-pool/
│   ├── Cargo.toml
│   └── src/bin/duckdb_pool.rs
├── pool_client.py
├── candle_cache.py
├── coin_graph.py
└── bin/bootstrap_duckdb_pool.sh
```

## Build

```bash
cd ./duckdb-pool
DUCKDB_LIB_DIR=$(brew --prefix duckdb)/lib cargo build --release --bin duckdb_pool
```

Or from repo root:

```bash
./bin/bootstrap_duckdb_pool.sh
```

## Runtime Contract

- Binary: `./duckdb-pool/target/release/duckdb_pool`
- Default socket: `/tmp/duckdb_pool.sock`
- Request: `{"query": "SELECT ..."}`
- Response: `{"rows": [[...]], "error": null}`

The pool supports stdin/stdout mode and Unix socket mode. Autotrade uses the
Unix socket path through [pool_client.py](/Users/jim/work/autotrade/pool_client.py).

## Integration Points

- [pool_client.py](/Users/jim/work/autotrade/pool_client.py): thin Python client
- [candle_cache.py](/Users/jim/work/autotrade/candle_cache.py): shared candle reads
- [coin_graph.py](/Users/jim/work/autotrade/coin_graph.py): graph loading path
- [run_training.sh](/Users/jim/work/autotrade/run_training.sh): byobu launcher for the pool + workers

## Why It Exists

DuckDB rejects mixed connection configurations against the same database file.
Running one local pool process keeps the database surface stable while letting
multiple Python workers query through the same connection owner.
