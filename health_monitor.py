#!/usr/bin/env python3
"""
Health Monitor - Lightweight HTTP status endpoint.

A minimal Flask service that provides health and metric summary endpoints
for monitoring the HRM system status.

Endpoints:
  GET /health          -> Health check (200 OK if system running)
  GET /metrics         -> Quick metrics summary
  GET /status          -> Detailed status with checkpoint info

Example:
  python3 health_monitor.py --port 8000
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify

app = Flask(__name__)

# Config from environment
DB_PATH = Path(os.getenv("DB_PATH", "candles.duckdb"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "./checkpoints"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))


def _get_latest_checkpoint() -> Dict:
    """Get the most recent checkpoint."""
    if not CHECKPOINT_DIR.exists():
        return None
    
    checkpoints = []
    for cp_path in CHECKPOINT_DIR.glob("*.pt"):
        try:
            stat = cp_path.stat()
            checkpoints.append({
                "name": cp_path.name,
                "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
            })
        except Exception:
            pass
    
    if not checkpoints:
        return None
    
    return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)[0]


def _get_active_workers() -> List[Dict]:
    """Get list of active workers from log files."""
    workers = []
    
    if not LOG_DIR.exists():
        return workers
    
    for log_file in LOG_DIR.glob("*.log"):
        try:
            stat = log_file.stat()
            workers.append({
                "name": log_file.stem,
                "last_update": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_kb": round(stat.st_size / 1024, 2),
            })
        except Exception:
            pass
    
    return workers


@app.route("/health")
def health():
    """Simple health check endpoint."""
    healthy = True
    checks = {
        "database": DB_PATH.exists(),
        "checkpoints_dir": CHECKPOINT_DIR.exists(),
        "logs_dir": LOG_DIR.exists(),
    }
    
    if not all(checks.values()):
        healthy = False
    
    status_code = 200 if healthy else 503
    
    return jsonify({
        "status": "healthy" if healthy else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
    }), status_code


@app.route("/metrics")
def metrics():
    """Quick metrics summary."""
    latest_cp = _get_latest_checkpoint()
    workers = _get_active_workers()
    
    # DB size
    db_size = 0
    if DB_PATH.exists():
        db_size = round(DB_PATH.stat().st_size / 1024 / 1024, 2)
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "database": {
            "path": str(DB_PATH),
            "exists": DB_PATH.exists(),
            "size_mb": db_size,
        },
        "checkpoints": {
            "dir": str(CHECKPOINT_DIR),
            "latest": latest_cp,
        },
        "workers": {
            "count": len(workers),
            "active": workers[:5],  # Top 5
        },
    })


@app.route("/status")
def status():
    """Detailed status endpoint."""
    return jsonify({
        "service": "health_monitor",
        "timestamp": datetime.now().isoformat(),
        "database": {
            "path": str(DB_PATH),
            "exists": DB_PATH.exists(),
            "size_mb": round(DB_PATH.stat().st_size / 1024 / 1024, 2) if DB_PATH.exists() else 0,
        },
        "checkpoint_dir": {
            "path": str(CHECKPOINT_DIR),
            "exists": CHECKPOINT_DIR.exists(),
        },
        "log_dir": {
            "path": str(LOG_DIR),
            "exists": LOG_DIR.exists(),
        },
    })


def main():
    parser = argparse.ArgumentParser(description="HRM Health Monitor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting Health Monitor on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
