# literbike-pool

DuckDB connection pool singleton bundled with autotrade.

## Lineage

Follows literbike tasktree patterns: superior IO via single connection
serving queries through stdin/stdout JSON protocol. Avoids DuckDB's
"multiple connection configuration conflicts" by maintaining one
singleton process.

## Build

```bash
cd literbike-pool
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
    ['./literbike-pool/target/release/duckdb_pool', './candles.duckdb'],
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
