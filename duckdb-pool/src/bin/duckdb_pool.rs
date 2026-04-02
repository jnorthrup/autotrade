//! DuckDB Connection Pool Server (Singleton)
//!
//! Bundled with autotrade as the local DuckDB pool process.
//! Superior IO pattern: single connection via stdin/stdout or Unix socket JSON protocol.
//!
//! Build:
//!   DUCKDB_LIB_DIR=$(brew --prefix duckdb)/lib cargo build --release
//!
//! Usage (stdin/stdout):
//!   ./target/release/duckdb_pool ./candles.duckdb
//!
//! Usage (Unix domain socket):
//!   ./target/release/duckdb_pool ./candles.duckdb --socket /tmp/duckdb.sock
//!
//! Protocol:
//!   Input:  {"query": "SELECT ..."}
//!   Output: {"rows": [[val1, val2, ...]], "error": null|"msg"}

use anyhow::Result;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::os::unix::net::UnixListener;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <db_path> [--socket <path>]", args[0]);
        eprintln!("  No --socket: stdin/stdout JSON protocol");
        eprintln!("  With --socket: Unix domain socket, multiple clients");
        std::process::exit(1);
    }

    let db_path = &args[1];

    // Parse optional --socket <path> from remaining args
    let socket_path: Option<String> = {
        let mut sp = None;
        let mut i = 2;
        while i < args.len() {
            if args[i] == "--socket" && i + 1 < args.len() {
                sp = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        }
        sp
    };

    eprintln!("[duckdb-pool] Starting for: {}", db_path);

    // Open single singleton connection
    let conn = duckdb::Connection::open(db_path)?;
    eprintln!("[duckdb-pool] Connection ready");

    let conn = Arc::new(Mutex::new(conn));

    match socket_path {
        Some(path) => {
            // Remove stale socket file if it exists
            let _ = fs::remove_file(&path);

            let listener = UnixListener::bind(&path)?;
            eprintln!("[duckdb-pool] Listening on socket: {}", path);

            // Accept connections in a loop, spawn a thread per connection
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let conn = Arc::clone(&conn);
                        let path_display = path.clone();
                        thread::spawn(move || {
                            eprintln!(
                                "[duckdb-pool] Client connected on socket: {}",
                                path_display
                            );
                            match stream.try_clone() {
                                Ok(cloned) => {
                                    let reader = BufReader::new(cloned);
                                    let writer = BufWriter::new(stream);
                                    handle_client(reader, writer, &conn);
                                }
                                Err(e) => {
                                    eprintln!("[duckdb-pool] try_clone failed: {}", e);
                                }
                            }
                            eprintln!(
                                "[duckdb-pool] Client disconnected on socket: {}",
                                path_display
                            );
                        });
                    }
                    Err(e) => {
                        eprintln!("[duckdb-pool] Accept error: {}", e);
                    }
                }
            }

            // Clean up socket on shutdown
            let _ = fs::remove_file(&path);
        }
        None => {
            // Original stdin/stdout mode
            let stdin = io::stdin();
            let stdout = io::stdout();
            let reader = BufReader::new(stdin.lock());
            let writer = BufWriter::new(stdout.lock());
            handle_client(reader, writer, &conn);
        }
    }

    Ok(())
}

/// Process JSON query lines from a reader, write responses to a writer.
/// Works for both stdin/stdout and Unix socket streams.
fn handle_client<R: BufRead, W: Write>(reader: R, mut writer: W, conn: &Arc<Mutex<duckdb::Connection>>) {
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let req: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                let resp = serde_json::json!({"error": format!("Invalid JSON: {}", e)});
                let _ = writeln!(writer, "{}", resp);
                let _ = writer.flush();
                continue;
            }
        };

        let query = match req.get("query") {
            Some(q) => q.as_str().unwrap_or(""),
            None => {
                let resp = serde_json::json!({"error": "Missing 'query' field"});
                let _ = writeln!(writer, "{}", resp);
                let _ = writer.flush();
                continue;
            }
        };

        let result = execute_query(conn, query);
        match serde_json::to_string(&result) {
            Ok(json) => {
                let _ = writeln!(writer, "{}", json);
                let _ = writer.flush();
            }
            Err(e) => {
                eprintln!("[duckdb-pool] Serialize error: {}", e);
            }
        }
    }
}

#[derive(serde::Serialize)]
struct QueryResponse {
    rows: Vec<Vec<serde_json::Value>>,
    error: Option<String>,
}

fn execute_query(conn: &Arc<Mutex<duckdb::Connection>>, query: &str) -> QueryResponse {
    let conn_guard = match conn.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };

    // Use raw_execute + raw_query to bypass the buggy Statement::query
    let mut stmt = match conn_guard.prepare(query) {
        Ok(s) => s,
        Err(e) => {
            return QueryResponse {
                rows: vec![],
                error: Some(format!("Prepare failed: {}", e)),
            };
        }
    };

    // Manually bind 0 params and execute
    if let Err(e) = stmt.execute([]) {
        return QueryResponse {
            rows: vec![],
            error: Some(format!("Execute failed: {}", e)),
        };
    };

    let column_count = stmt.column_count();
    let mut rows: Vec<Vec<serde_json::Value>> = vec![];

    // Use raw_query which reads from the already-executed statement
    let mut result_iter = stmt.raw_query();
    while let Ok(Some(row)) = result_iter.next() {
        let mut row_vals = vec![];
        for col_idx in 0..column_count {
            row_vals.push(get_json_value(&row, col_idx));
        }
        rows.push(row_vals);
    }

    QueryResponse { rows, error: None }
}

fn get_json_value(row: &duckdb::Row, col_idx: usize) -> serde_json::Value {
    use base64::prelude::*;
    use duckdb::types::ValueRef;

    let value_ref = match row.get_ref(col_idx) {
        Ok(v) => v,
        Err(_) => return serde_json::json!(null),
    };

    match value_ref {
        ValueRef::Null => serde_json::json!(null),
        ValueRef::Boolean(b) => serde_json::json!(b),
        ValueRef::TinyInt(i) => serde_json::json!(i),
        ValueRef::SmallInt(i) => serde_json::json!(i),
        ValueRef::Int(i) => serde_json::json!(i),
        ValueRef::BigInt(i) => serde_json::json!(i),
        ValueRef::HugeInt(i) => serde_json::json!(i),
        ValueRef::UTinyInt(u) => serde_json::json!(u),
        ValueRef::USmallInt(u) => serde_json::json!(u),
        ValueRef::UInt(u) => serde_json::json!(u),
        ValueRef::UBigInt(u) => serde_json::json!(u),
        ValueRef::Float(f) => serde_json::json!(f),
        ValueRef::Double(d) => serde_json::json!(d),
        ValueRef::Timestamp(unit, value) => {
            // Convert DuckDB timestamp to ISO 8601 string
            let micros = match unit {
                duckdb::types::TimeUnit::Microsecond => value,
                duckdb::types::TimeUnit::Millisecond => value * 1000,
                duckdb::types::TimeUnit::Nanosecond => value / 1000,
                _ => value,
            };
            let secs = micros / 1_000_000;
            let frac = micros % 1_000_000;
            match chrono::NaiveDateTime::from_timestamp_opt(secs, (frac * 1000) as u32) {
                Some(dt) => serde_json::json!(dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()),
                None => serde_json::json!(format!("{:?}", value)),
            }
        }
        ValueRef::Text(s) => match std::str::from_utf8(s) {
            Ok(text) => serde_json::json!(text),
            Err(_) => serde_json::json!(null),
        },
        ValueRef::Blob(b) => serde_json::json!(BASE64_STANDARD.encode(b)),
        _ => serde_json::json!(null),
    }
}
