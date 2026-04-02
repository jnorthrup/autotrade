"""
PoolClient -- talks to the local autotrade duckdb_pool via unix domain socket.

All training processes share one pool server, which holds the single
DuckDB connection.  No more lock conflicts.

The pool protocol is: {"query": "SQL"} -> {"rows": [[...]], "error": null}
No server-side param binding, so we interpolate client-side.
"""

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import numpy as np

DEFAULT_SOCKET = "/tmp/duckdb_pool.sock"
POOL_BINARY = (
    Path(__file__).parent / "duckdb-pool" / "target" / "release" / "duckdb_pool"
)
POOL_AUTOSTART_LOG = Path(__file__).parent / "logs" / "duckdb_pool.autostart.log"

_pool_bootstrap_lock = threading.Lock()
_pool_ready = threading.Event()
_pool_process = None
_pool_log_handle = None


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


def _open_pool_log():
    global _pool_log_handle
    if _pool_log_handle is None or _pool_log_handle.closed:
        POOL_AUTOSTART_LOG.parent.mkdir(parents=True, exist_ok=True)
        _pool_log_handle = open(POOL_AUTOSTART_LOG, "ab", buffering=0)
    return _pool_log_handle


def _start_pool_process(db_path: str, socket_path: str):
    return subprocess.Popen(
        [str(POOL_BINARY), db_path, "--socket", socket_path],
        cwd=str(Path(__file__).parent),
        stdin=subprocess.DEVNULL,
        stdout=_open_pool_log(),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )


class PoolClient:
    """Thin client for the local DuckDB pool server."""

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


def ensure_pool_running(
    db_path: str,
    socket_path: Optional[str] = None,
    startup_timeout: float = 10.0,
) -> PoolClient:
    """Start the local DuckDB pool once and wait until it is responsive."""
    sp = socket_path or DEFAULT_SOCKET
    if pool_is_running(sp):
        _pool_ready.set()
        return PoolClient(sp)

    resolved_db_path = str(Path(db_path).resolve())
    if not POOL_BINARY.exists():
        raise RuntimeError(
            f"DuckDB pool binary not found at {POOL_BINARY}. "
            "Build it with ./bin/bootstrap_duckdb_pool.sh"
        )

    with _pool_bootstrap_lock:
        global _pool_process

        if pool_is_running(sp):
            _pool_ready.set()
            return PoolClient(sp)

        if os.path.exists(sp):
            try:
                os.remove(sp)
            except OSError:
                pass

        if _pool_process is None or _pool_process.poll() is not None:
            _pool_process = _start_pool_process(resolved_db_path, sp)
        proc = _pool_process

    deadline = time.monotonic() + max(1.0, startup_timeout)
    while time.monotonic() < deadline:
        if pool_is_running(sp):
            _pool_ready.set()
            return PoolClient(sp)
        if proc is not None and proc.poll() is not None:
            break
        time.sleep(0.1)

    _pool_ready.clear()
    exit_note = ""
    if proc is not None and proc.poll() is not None:
        exit_note = f" exited with code {proc.returncode}"
    raise RuntimeError(
        f"DuckDB pool failed to start for {resolved_db_path} on {sp}{exit_note}. "
        f"See {POOL_AUTOSTART_LOG}."
    )
