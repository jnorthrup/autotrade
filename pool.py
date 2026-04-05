"""pool.py - Simplest possible threaded DuckDB connection pool."""

import threading
import queue
import duckdb
from pathlib import Path
from typing import Optional, List, Any


class DuckDBPool:
    """Minimal threaded connection pool for DuckDB."""
    
    def __init__(self, db_path: str, pool_size: int = 4):
        self.db_path = str(Path(db_path).resolve())
        self._pool: queue.Queue = queue.Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 0
        self._max_size = pool_size
        
    def _create_conn(self) -> duckdb.DuckDBPyConnection:
        """Create a new read-only connection."""
        return duckdb.connect(self.db_path, read_only=True)
    
    def get(self) -> duckdb.DuckDBPyConnection:
        """Get a connection from the pool."""
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._created < self._max_size:
                    self._created += 1
                    return self._create_conn()
            return self._pool.get()
    
    def put(self, conn: duckdb.DuckDBPyConnection):
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            conn.close()
    
    def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[List[Any]]:
        """Execute a query and return rows."""
        conn = self.get()
        try:
            if params:
                result = conn.execute(sql, params).fetchall()
            else:
                result = conn.execute(sql).fetchall()
            return result
        finally:
            self.put(conn)
    
    def execute_df(self, sql: str, params: Optional[List[Any]] = None):
        """Execute a query and return DataFrame."""
        conn = self.get()
        try:
            if params:
                return conn.execute(sql, params).df()
            return conn.execute(sql).df()
        finally:
            self.put(conn)
    
    def close(self):
        """Close all connections."""
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break


# Global pool instance
_pool: Optional[DuckDBPool] = None
_pool_lock = threading.Lock()


def get_pool(db_path: Optional[str] = None) -> DuckDBPool:
    """Get or create the global pool."""
    global _pool
    with _pool_lock:
        if _pool is None:
            from cache import Config
            _pool = DuckDBPool(str(Config.DB_PATH) if db_path is None else db_path)
        return _pool


def close_pool():
    """Close the global pool."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
