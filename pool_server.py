#!/usr/bin/env python3
import socket
import json
import sys
import duckdb
import threading
import queue
import datetime
import os

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

def handle_client(conn_socket, db_queue):
    conn_socket.settimeout(60)
    buffer = ""
    try:
        while True:
            data = conn_socket.recv(65536)
            if not data:
                break
            buffer += data.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                
                res_q = queue.Queue()
                db_queue.put((req.get("query"), res_q))
                res = res_q.get()
                
                conn_socket.sendall((json.dumps(res, cls=CustomEncoder) + "\n").encode("utf-8"))
    except Exception:
        pass
    finally:
        conn_socket.close()

def db_worker(db_path, task_queue):
    db = duckdb.connect(db_path, read_only=False)
    while True:
        try:
            sql, res_q = task_queue.get()
            if sql is None:
                break
            cursor = db.cursor()
            try:
                cursor.execute(sql)
                rows = cursor.fetchall() if cursor.description else []
                res_q.put({"rows": [list(r) for r in rows]})
            except Exception as e:
                res_q.put({"error": str(e)})
            finally:
                cursor.close()
        except Exception as e:
            res_q.put({"error": f"Worker error: {e}"})

def main():
    if len(sys.argv) < 2:
        print("Usage: pool_server.py <db_path> [--socket <socket_path>]")
        sys.exit(1)
        
    db_path = sys.argv[1]
    socket_path = "/tmp/duckdb_pool.sock"
    if "--socket" in sys.argv:
        idx = sys.argv.index("--socket")
        if idx + 1 < len(sys.argv):
            socket_path = sys.argv[idx + 1]

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(128)
    
    print(f"DuckDB Pool Server running on {socket_path} for {db_path}...")
    
    db_queue = queue.Queue()
    db_thread = threading.Thread(target=db_worker, args=(db_path, db_queue), daemon=True)
    db_thread.start()

    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle_client, args=(conn, db_queue), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.remove(socket_path)

if __name__ == "__main__":
    main()
