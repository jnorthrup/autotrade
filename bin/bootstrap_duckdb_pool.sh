#!/usr/bin/env bash
# Build the local autotrade duckdb_pool binary.

set -euo pipefail

AUTOTRADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POOL_ROOT="$AUTOTRADE_ROOT/duckdb-pool"

echo "[bootstrap] Building local duckdb_pool..."
echo "[bootstrap] AUTOTRADE_ROOT: $AUTOTRADE_ROOT"
echo "[bootstrap] POOL_ROOT: $POOL_ROOT"

if [[ ! -d "$POOL_ROOT" ]]; then
    echo "[bootstrap] ERROR: duckdb-pool crate not found at $POOL_ROOT"
    exit 1
fi

cd "$POOL_ROOT"

# Build with DuckDB library path
DUCKDB_LIB_DIR="${DUCKDB_LIB_DIR:-$(brew --prefix duckdb)/lib}"

echo "[bootstrap] DUCKDB_LIB_DIR: $DUCKDB_LIB_DIR"

if [[ ! -d "$DUCKDB_LIB_DIR/../include" ]]; then
    echo "[bootstrap] ERROR: DuckDB not found via brew"
    echo "[bootstrap] Install: brew install duckdb"
    exit 1
fi

# Build the pool server binary
echo "[bootstrap] Compiling duckdb_pool..."
DUCKDB_LIB_DIR="$DUCKDB_LIB_DIR" cargo build --release --bin duckdb_pool

BINARY="$POOL_ROOT/target/release/duckdb_pool"

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
echo "[bootstrap] SUCCESS: local duckdb pool ready"
echo "[bootstrap] Use from autotrade:"
echo "[bootstrap]   import subprocess"
echo "[bootstrap]   proc = subprocess.Popen(['$LINK', './candles.duckdb'], ...)"
