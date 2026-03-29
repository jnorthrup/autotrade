//! DuckDB Connection Pool Server (Singleton)
//!
//! Bundled with autotrade following literbike tasktree lineage.
//! Superior IO pattern: single connection via stdin/stdout JSON protocol.
//!
//! Build:
//!   DUCKDB_LIB_DIR=$(brew --prefix duckdb)/lib cargo build --release
//!
//! Usage:
//!   ./target/release/duckdb_pool ./candles.duckdb
//!
//! Protocol:
//!   Input (stdin):  {"query": "SELECT ..."}
//!   Output (stdout): {"rows": [[val1, val2, ...]], "error": null|"msg"}

use anyhow::Result;
use std::env;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::sync::{Arc, Mutex};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <db_path>", args[0]);
        eprintln!("Singleton DuckDB pool server - stdin/stdout JSON protocol");
        std::process::exit(1);
    }

    let db_path = &args[1];
    eprintln!("[literbike-pool] Starting for: {}", db_path);

    // Open single singleton connection
    let conn = duckdb::Connection::open(db_path)?;
    eprintln!("[literbike-pool] Connection ready");

    let conn = Arc::new(Mutex::new(conn));

    let stdin = io::stdin();
    let stdout = io::stdout();
    let reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());

    // Process JSON query lines
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
                writeln!(writer, "{}", resp)?;
                writer.flush()?;
                continue;
            }
        };

        let query = match req.get("query") {
            Some(q) => q.as_str().unwrap_or(""),
            None => {
                let resp = serde_json::json!({"error": "Missing 'query' field"});
                writeln!(writer, "{}", resp)?;
                writer.flush()?;
                continue;
            }
        };

        let result = execute_query(&conn, query);
        writeln!(writer, "{}", serde_json::to_string(&result)?)?;
        writer.flush()?;
    }

    Ok(())
}

#[derive(serde::Serialize)]
struct QueryResponse {
    rows: Vec<Vec<serde_json::Value>>,
    error: Option<String>,
}

fn execute_query(conn: &Arc<Mutex<duckdb::Connection>>, query: &str) -> QueryResponse {
    let conn_guard = conn.lock().unwrap();
    let mut rows = vec![];

    let mut stmt = match conn_guard.prepare(query) {
        Ok(s) => s,
        Err(e) => return QueryResponse {
            rows: vec![],
            error: Some(format!("Prepare failed: {}", e)),
        }
    };

    let column_count = stmt.column_count();
    let mut result = match stmt.query([]) {
        Ok(r) => r,
        Err(e) => return QueryResponse {
            rows: vec![],
            error: Some(format!("Query failed: {}", e)),
        }
    };

    while let Ok(Some(row)) = result.next() {
        let mut row_vals = vec![];
        for col_idx in 0..column_count {
            row_vals.push(get_json_value(&row, col_idx));
        }
        rows.push(row_vals);
    }

    QueryResponse { rows, error: None }
}

fn get_json_value(row: &duckdb::Row, col_idx: usize) -> serde_json::Value {
    use duckdb::types::ValueRef;
    use base64::prelude::*;

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
        ValueRef::Timestamp(s, _) => serde_json::json!(format!("{:?}", s)),
        ValueRef::Text(s) => match std::str::from_utf8(s) {
            Ok(text) => serde_json::json!(text),
            Err(_) => serde_json::json!(null),
        },
        ValueRef::Blob(b) => serde_json::json!(BASE64_STANDARD.encode(b)),
        _ => serde_json::json!(null),
    }
}
