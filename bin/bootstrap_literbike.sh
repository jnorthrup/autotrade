#!/usr/bin/env bash
# Bootstrap literbike duckdb_pool_server for autotrade
# This script builds the superior IO backend that autotrade depends on

set -euo pipefail

LITERBIKE_ROOT="${LITERBIKE_ROOT:-$HOME/work/literbike}"
AUTOTRADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[bootstrap] Building literbike duckdb_pool_server..."
echo "[bootstrap] LITERBIKE_ROOT: $LITERBIKE_ROOT"
echo "[bootstrap] AUTOTRADE_ROOT: $AUTOTRADE_ROOT"

if [[ ! -d "$LITERBIKE_ROOT" ]]; then
    echo "[bootstrap] ERROR: literbike not found at $LITERBIKE_ROOT"
    exit 1
fi

cd "$LITERBIKE_ROOT"

# Build with DuckDB library path
DUCKDB_LIB_DIR=$(brew --prefix duckdb)/lib

echo "[bootstrap] DUCKDB_LIB_DIR: $DUCKDB_LIB_DIR"

if [[ ! -d "$DUCKDB_LIB_DIR/../include" ]]; then
    echo "[bootstrap] ERROR: DuckDB not found via brew"
    echo "[bootstrap] Install: brew install duckdb"
    exit 1
fi

# Build the pool server binary
echo "[bootstrap] Compiling duckdb_pool_server..."
DUCKDB_LIB_DIR="$DUCKDB_LIB_DIR" cargo build --release --bin duckdb_pool_server

BINARY="$LITERBIKE_ROOT/target/release/duckdb_pool_server"

if [[ ! -f "$BINARY" ]]; then
    echo "[bootstrap] ERROR: Binary not built at $BINARY"
    exit 1
fi

echo "[bootstrap] ✓ Built: $BINARY"
echo "[bootstrap] Size: $(ls -lh "$BINARY" | awk '{print $5}')"

# Create symlink in autotrade bin for convenience
LINK="$AUTOTRADE_ROOT/bin/duckdb_pool"
if [[ -L "$LINK" ]]; then
    rm "$LINK"
fi
ln -s "$BINARY" "$LINK"
echo "[bootstrap] ✓ Symlinked: $LINK → $BINARY"

# Quick smoke test
echo "[bootstrap] Running smoke test..."
echo '{"query": "SELECT 1"}' | "$BINARY" ":memory:" | grep -q '"rows"'
if [[ $? -eq 0 ]]; then
    echo "[bootstrap] ✓ Smoke test passed"
else
    echo "[bootstrap] ✗ Smoke test failed"
    exit 1
fi

echo ""
echo "[bootstrap] SUCCESS: literbike pool server ready"
echo "[bootstrap] Use from autotrade:"
echo "[bootstrap]   import subprocess"
echo "[bootstrap]   proc = subprocess.Popen(['$LINK', './candles.duckdb'], ...)"
