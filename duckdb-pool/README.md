# duckdb-pool

DuckDB connection pool singleton bundled with autotrade.

The pool is owned locally in this repo. It serves queries through a
single DuckDB connection over stdin/stdout or a Unix socket JSON
protocol to avoid DuckDB "multiple connection configuration conflicts".

## Build

```bash
cd duckdb-pool
DUCKDB_LIB_DIR=$(brew --prefix duckdb)/lib cargo build --release
```

Binary: `./target/release/duckdb_pool`

## Usage

```bash
./target/release/duckdb_pool ./candles.duckdb
```

## Protocol

**Input (stdin, line-delimited JSON):**
```json
{"query": "SELECT DISTINCT product_id FROM candles"}
```

**Output (stdout, line-delimited JSON):**
```json
{"rows": [["BTC-USD"], ["ETH-USD"], ...], "error": null}
```

## Python Integration

```python
import subprocess, json

proc = subprocess.Popen(
    ['./duckdb-pool/target/release/duckdb_pool', './candles.duckdb'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

# Execute query
request = json.dumps({"query": "SELECT * FROM candles LIMIT 1"})
proc.stdin.write(request + "\n")
proc.stdin.flush()

# Read response
response = json.loads(proc.stdout.readline())
print(response)
```

## Why This Pattern

1. **Single connection**: No configuration conflicts
2. **Process isolation**: Crash doesn't kill training
3. **Simple protocol**: JSON over stdin/stdout
4. **Language agnostic**: Works with Python, Rust, anything
