"""
PoolClient -- talks to literbike duckdb_pool via unix domain socket.

All training processes share one pool server, which holds the single
DuckDB connection.  No more lock conflicts.

The pool protocol is: {"query": "SQL"} -> {"rows": [[...]], "error": null}
No server-side param binding, so we interpolate client-side.
"""

import json
import os
import socket
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import numpy as np

DEFAULT_SOCKET = "/tmp/duckdb_pool.sock"
POOL_BINARY = (
    Path(__file__).parent / "literbike-pool" / "target" / "release" / "duckdb_pool"
)


def _sql_literal(val) -> str:
    """Convert a Python value to a SQL literal for client-side interpolation."""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        return repr(float(val))
    if isinstance(val, (pd.Timestamp,)):
        return f"'{val.isoformat()}'"
    if isinstance(val, str):
        escaped = val.replace("'", "''")
        return f"'{escaped}'"
    if hasattr(val, 'isoformat'):
        return f"'{val.isoformat()}'"
    return f"'{val}'"


def _interpolate_params(sql: str, params: Optional[List[Any]] = None) -> str:
    """Replace ? placeholders with interpolated SQL literals."""
    if not params:
        return sql
    parts = sql.split("?")
    result = parts[0]
    for i, part in enumerate(parts[1:], 0):
        if i < len(params):
            result += _sql_literal(params[i]) + part
        else:
            result += "?" + part
    return result


class PoolClient:
    """Thin client for the literbike DuckDB pool server."""

    def __init__(self, socket_path: Optional[str] = None):
        self.socket_path = socket_path or DEFAULT_SOCKET

    def _call(self, req: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
            sock.settimeout(30)
            sock.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            line = buf.split(b"\n", 1)[0]
            return json.loads(line)
        finally:
            sock.close()

    def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[List[Any]]:
        """Execute a query, return rows as list of lists."""
        interpolated = _interpolate_params(sql, params)
        req = {"query": interpolated}
        resp = self._call(req)
        if resp.get("error"):
            raise RuntimeError(f"Pool error: {resp['error']}")
        return resp.get("rows", [])

    def execute_df(self, sql: str, params: Optional[List[Any]] = None):
        """Execute a query, return a pandas DataFrame."""
        rows = self.execute(sql, params)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def ping(self) -> bool:
        """Check if pool server is alive."""
        try:
            result = self.execute("SELECT 1")
            return result == [[1]]
        except Exception:
            return False


def pool_is_running(socket_path: Optional[str] = None) -> bool:
    """Quick check if the pool server socket exists and is responsive."""
    sp = socket_path or DEFAULT_SOCKET
    if not os.path.exists(sp):
        return False
    return PoolClient(sp).ping()
