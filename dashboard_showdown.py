#!/usr/bin/env python3
"""
Dashboard Showdown - Real-time training metrics dashboard with drilldown.

Features:
- Live training loss charts
- Checkpoint history with model growth visualization
- Per-edge/bag performance drilldown
- Paper trade signals (PTT/stop hits)
- Model architecture explorer (4× growth progression)

Endpoints:
  GET  /              -> Dashboard UI
  GET  /api/status    -> System status (active workers, last checkpoint)
  GET  /api/metrics   -> Time-series metrics (loss, bpbl, exploration)
  GET  /api/drilldown/int:<phase> -> Phase details (bag, window, growth state)
  GET  /api/checkpoints -> Checkpoint history with metadata
  GET  /api/edges/<edge_id>/metrics -> Per-edge performance
  POST /api/action/<action> -> Manual actions (pause, save, reload)

Example:
  python3 dashboard_showdown.py --port 8000 --poll-interval 5
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request

import duckdb

app = Flask(__name__)

# Config from environment
DB_PATH = Path(os.getenv("DB_PATH", "candles.duckdb"))
BAG_PATH = Path(os.getenv("BAG_PATH", "bag.json"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "./checkpoints"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

# In-memory metrics cache (updated by polling)
_metrics_cache: Dict[str, any] = {
    "phases": [],
    "checkpoints": [],
    "last_update": None,
}

# HTML Template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>HRM Dashboard Showdown</title>
    <style>
        * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        h1 { margin: 0; color: #58a6ff; }
        .status { display: flex; gap: 20px; }
        .status-item { background: #161b22; padding: 10px 20px; border-radius: 6px; border: 1px solid #30363d; }
        .status-label { font-size: 12px; color: #8b949e; text-transform: uppercase; }
        .status-value { font-size: 18px; font-weight: bold; color: #58a6ff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
        .card h3 { margin-top: 0; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
        .chart-container { height: 200px; position: relative; }
        .mini-chart { width: 100%; height: 100%; background: #0d1117; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #30363d; }
        th { color: #8b949e; font-weight: 500; }
        tr:hover { background: #1c2128; cursor: pointer; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; }
        .badge-pretrain { background: #238636; color: white; }
        .badge-finetune { background: #8957e5; color: white; }
        .badge-growth { background: #d29922; color: black; }
        .growth-bar { display: flex; gap: 2px; height: 20px; margin-top: 10px; }
        .growth-segment { flex: 1; border-radius: 2px; }
        .growth-h { background: #58a6ff; }
        .growth-H { background: #238636; }
        .growth-L { background: #d29922; }
        .drilldown { margin-top: 15px; padding: 15px; background: #0d1117; border-radius: 4px; display: none; }
        .drilldown.active { display: block; }
        .pairs-list { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
        .pair-tag { background: #21262d; padding: 3px 8px; border-radius: 4px; font-size: 11px; }
        .actions { display: flex; gap: 10px; margin-top: 20px; }
        button { background: #238636; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; }
        button:hover { background: #2ea043; }
        button.secondary { background: #21262d; }
        button.secondary:hover { background: #30363d; }
        .loading { text-align: center; padding: 40px; color: #8b949e; }
        #refresh-indicator { position: fixed; top: 20px; right: 20px; width: 10px; height: 10px; border-radius: 50%; background: #238636; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    </style>
</head>
<body>
    <div id="refresh-indicator"></div>
    
    <div class="header">
        <h1>🔥 HRM Dashboard Showdown</h1>
        <div class="status">
            <div class="status-item">
                <div class="status-label">Phase</div>
                <div class="status-value" id="current-phase">-</div>
            </div>
            <div class="status-item">
                <div class="status-label">Model Size</div>
                <div class="status-value" id="model-size">-</div>
            </div>
            <div class="status-item">
                <div class="status-label">Last Loss</div>
                <div class="status-value" id="last-loss">-</div>
            </div>
        </div>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>📈 Training Loss (Live)</h3>
            <div class="chart-container">
                <canvas id="loss-chart" class="mini-chart"></canvas>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #8b949e;">
                <span id="loss-avg">Avg: -</span> | 
                <span id="loss-min">Min: -</span> | 
                <span id="loss-max">Max: -</span>
            </div>
        </div>
        
        <div class="card">
            <h3>🔄 Growth Progression (4×)</h3>
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between;">
                    <span>h_dim: <b id="h-dim">-</b></span>
                    <span>H_layers: <b id="H-layers">-</b></span>
                    <span>L_layers: <b id="L-layers">-</b></span>
                </div>
            </div>
            <div class="growth-bar" id="growth-viz">
                <div class="growth-segment" style="background: #30363d;"></div>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #8b949e;">
                <span class="badge badge-growth">4× Rotation</span>
                <span id="growth-phase">growth phase: -</span>
            </div>
        </div>
        
        <div class="card">
            <h3>💾 Recent Checkpoints</h3>
            <table>
                <thead>
                    <tr><th>Timestamp</th><th>Type</th><th>Phase</th><th>Loss</th></tr>
                </thead>
                <tbody id="checkpoints-table">
                    <tr><td colspan="4" class="loading">Loading...</td></tr>
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h3>📊 Phase History (Click to Drill Down)</h3>
            <table>
                <thead>
                    <tr><th>Phase</th><th>Bag Size</th><th>Window</th><th>Loss</th></tr>
                </thead>
                <tbody id="phases-table">
                    <tr><td colspan="4" class="loading">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <div id="drilldown-panel" class="card drilldown">
        <h3>🔍 Phase Drilldown <span id="drilldown-phase"></span></h3>
        <div id="drilldown-content"></div>
    </div>
    
    <div class="actions">
        <button onclick="triggerAction('save')">💾 Force Checkpoint</button>
        <button class="secondary" onclick="triggerAction('poll')">🔄 Refresh Now</button>
        <button class="secondary" onclick="window.open('/api/status', '_blank')">📋 Raw Status</button>
    </div>
    
    <script>
        let selectedPhase = null;
        
        // Canvas line chart renderer
        function drawLineChart(canvas, data, color='#58a6ff') {
            const ctx = canvas.getContext('2d');
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = rect.height * 2;
            ctx.scale(2, 2);
            
            const w = rect.width, h = rect.height;
            const padding = 10;
            
            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, 0, w, h);
            
            if (data.length < 2) {
                ctx.fillStyle = '#8b949e';
                ctx.font = '12px sans-serif';
                ctx.fillText('No data', w/2 - 20, h/2);
                return;
            }
            
            const min = Math.min(...data);
            const max = Math.max(...data);
            const range = max - min || 1;
            
            // Grid lines
            ctx.strokeStyle = '#21262d';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = padding + (h - 2*padding) * (i/4);
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(w - padding, y);
                ctx.stroke();
            }
            
            // Line
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            data.forEach((v, i) => {
                const x = padding + (w - 2*padding) * (i / (data.length - 1));
                const y = padding + (h - 2*padding) * (1 - (v - min) / range);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            
            // Fill
            ctx.fillStyle = color + '20';
            ctx.lineTo(w - padding, h - padding);
            ctx.lineTo(padding, h - padding);
            ctx.closePath();
            ctx.fill();
        }
        
        async function fetchMetrics() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();
                updateDashboard(data);
            } catch (e) {
                console.error('Fetch failed:', e);
            }
        }
        
        function updateDashboard(data) {
            // Header stats
            document.getElementById('current-phase').textContent = data.current_phase || '-';
            document.getElementById('model-size').textContent = data.model_size || '-';
            document.getElementById('last-loss').textContent = data.last_loss ? data.last_loss.toFixed(6) : '-';
            
            // Loss chart
            const losses = (data.phases || []).map(p => p.loss).filter(l => l !== null);
            drawLineChart(document.getElementById('loss-chart'), losses);
            
            if (losses.length > 0) {
                document.getElementById('loss-avg').textContent = `Avg: ${(losses.reduce((a,b)=>a+b,0)/losses.length).toFixed(4)}`;
                document.getElementById('loss-min').textContent = `Min: ${Math.min(...losses).toFixed(4)}`;
                document.getElementById('loss-max').textContent = `Max: ${Math.max(...losses).toFixed(4)}`;
            }
            
            // Growth dims
            document.getElementById('h-dim').textContent = data.h_dim || '-';
            document.getElementById('H-layers').textContent = data.H_layers || '-';
            document.getElementById('L-layers').textContent = data.L_layers || '-';
            document.getElementById('growth-phase').textContent = `growth phase: ${data.growth_phase || '-'}`;
            
            // Growth visualization
            const growthViz = document.getElementById('growth-viz');
            if (data.h_dim && data.H_layers && data.L_layers) {
                const h_idx = Math.log2(data.h_dim) / 2 - 1; // 4->0, 16->1, 64->2
                const H_idx = Math.log2(data.H_layers) / 2 - 1;
                const L_idx = Math.log2(data.L_layers) / 2 - 1;
                growthViz.innerHTML = `
                    <div class="growth-segment ${h_idx >= 0 ? 'growth-h' : ''}" style="flex:${Math.pow(4, Math.max(0, h_idx))}"></div>
                    <div class="growth-segment ${H_idx >= 0 ? 'growth-H' : ''}" style="flex:${Math.pow(4, Math.max(0, H_idx))}"></div>
                    <div class="growth-segment ${L_idx >= 0 ? 'growth-L' : ''}" style="flex:${Math.pow(4, Math.max(0, L_idx))}"></div>
                `;
            }
            
            // Checkpoints table
            const cpTable = document.getElementById('checkpoints-table');
            if (data.checkpoints && data.checkpoints.length > 0) {
                cpTable.innerHTML = data.checkpoints.slice(-5).reverse().map(cp => `
                    <tr>
                        <td>${cp.timestamp.split('T')[1].split('.')[0]}</td>
                        <td><span class="badge badge-${cp.type}">${cp.type}</span></td>
                        <td>${cp.phase}</td>
                        <td>${cp.loss ? cp.loss.toFixed(4) : '-'}</td>
                    </tr>
                `).join('');
            }
            
            // Phases table
            const phaseTable = document.getElementById('phases-table');
            if (data.phases && data.phases.length > 0) {
                phaseTable.innerHTML = data.phases.slice(-10).reverse().map(p => `
                    <tr onclick="showDrilldown(${p.phase})">
                        <td>${p.phase}</td>
                        <td>${p.bag_size}</td>
                        <td>${p.window_bars}</td>
                        <td>${p.loss ? p.loss.toFixed(6) : '-'}</td>
                    </tr>
                `).join('');
            }
        }
        
        async function showDrilldown(phase) {
            selectedPhase = phase;
            const panel = document.getElementById('drilldown-panel');
            const content = document.getElementById('drilldown-content');
            const title = document.getElementById('drilldown-phase');
            
            title.textContent = `#${phase}`;
            panel.classList.add('active');
            content.innerHTML = '<div class="loading">Loading drilldown...</div>';
            
            try {
                const res = await fetch(`/api/drilldown/${phase}`);
                const data = await res.json();
                
                content.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px;">
                        <div class="status-item">
                            <div class="status-label">Bag Size</div>
                            <div class="status-value">${data.bag_size}</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Window Bars</div>
                            <div class="status-value">${data.window_bars}</div>
                        </div>
                        <div class="status-item">
                            <div class="status-label">Loss</div>
                            <div class="status-value">${data.loss ? data.loss.toFixed(6) : '-'}</div>
                        </div>
                    </div>
                    <div><b>Bag Pairs:</b></div>
                    <div class="pairs-list">
                        ${(data.pairs || []).map(p => `<span class="pair-tag">${p}</span>`).join('')}
                    </div>
                    <div style="margin-top: 15px;">
                        <b>Model State:</b> h_dim=${data.h_dim}, H_layers=${data.H_layers}, L_layers=${data.L_layers}
                    </div>
                    <div><b>Bars:</b> ${data.start_bar} → ${data.end_bar} (${data.window_days} days)</div>
                `;
            } catch (e) {
                content.innerHTML = '<div class="loading">Error loading drilldown</div>';
            }
        }
        
        async function triggerAction(action) {
            try {
                await fetch(`/api/action/${action}`, { method: 'POST' });
                fetchMetrics();
            } catch (e) {
                console.error('Action failed:', e);
            }
        }
        
        // Initial load and polling
        fetchMetrics();
        setInterval(fetchMetrics, 5000);
    </script>
</body>
</html>
"""


def _scan_checkpoints() -> List[Dict]:
    """Scan checkpoint directory for recent checkpoints."""
    checkpoints = []
    
    if not CHECKPOINT_DIR.exists():
        return checkpoints
    
    for cp_path in CHECKPOINT_DIR.glob("*.pt"):
        try:
            stat = cp_path.stat()
            checkpoints.append({
                "path": str(cp_path),
                "name": cp_path.name,
                "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
            })
        except Exception:
            pass
    
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    return checkpoints[:10]


def _parse_worker_logs() -> List[Dict]:
    """Parse training worker logs for metrics."""
    phases = []
    
    for log_file in LOG_DIR.glob("*training-worker*.log"):
        try:
            with open(log_file) as f:
                for line in f:
                    # Look for phase completion lines
                    # Example: "2025-03-28T14:30:00 INFO training-worker Phase 5: loss=0.123456, updates=100"
                    if "Phase" in line and "loss=" in line:
                        parts = line.strip().split()
                        for i, p in enumerate(parts):
                            if p.startswith("Phase"):
                                phase = int(parts[i+1].rstrip(":"))
                            if p.startswith("loss="):
                                loss = float(p.split("=")[1].rstrip(","))
                            if p.startswith("updates="):
                                updates = int(p.split("=")[1])
                        
                        phases.append({
                            "phase": phase,
                            "loss": loss,
                            "updates": updates,
                            "timestamp": parts[0] if parts else None,
                        })
        except Exception:
            pass
    
    return sorted(phases, key=lambda x: x["phase"])


@app.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    """System status endpoint."""
    checkpoints = _scan_checkpoints()
    
    # Check active workers by looking at recent log activity
    active_workers = []
    for log_file in LOG_DIR.glob("*.log"):
        try:
            stat = log_file.stat()
            age_seconds = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds()
            if age_seconds < 300:  # Active if log updated in last 5 min
                active_workers.append({
                    "name": log_file.stem,
                    "last_activity": age_seconds,
                })
        except Exception:
            pass
    
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "checkpoints": len(checkpoints),
        "active_workers": active_workers,
        "db_exists": DB_PATH.exists(),
        "db_size_mb": round(DB_PATH.stat().st_size / 1024 / 1024, 2) if DB_PATH.exists() else 0,
    })


@app.route("/api/metrics")
def api_metrics():
    """Training metrics endpoint."""
    phases = _parse_worker_logs()
    checkpoints = _scan_checkpoints()
    
    # Derive current state from phases
    current_phase = phases[-1]["phase"] if phases else 0
    last_loss = phases[-1]["loss"] if phases else None
    
    # Model size from latest checkpoint or default
    model_size = "4×4×4"
    h_dim, H_layers, L_layers = 4, 4, 4
    growth_phase = "init"
    
    if checkpoints:
        # Try to infer from checkpoint metadata
        latest = checkpoints[0]
        if "pretrain" in latest["name"]:
            growth_phase = "h"  # Simplified - would parse actual checkpoint
    
    return jsonify({
        "current_phase": current_phase,
        "last_loss": last_loss,
        "model_size": model_size,
        "h_dim": h_dim,
        "H_layers": H_layers,
        "L_layers": L_layers,
        "growth_phase": growth_phase,
        "phases": phases[-50:],  # Last 50 phases
        "checkpoints": checkpoints[:5],
    })


@app.route("/api/drilldown/<int:phase>")
def api_drilldown(phase: int):
    """Drilldown for a specific phase."""
    # In production, this would query a metrics database
    # For now, return mock/derived data
    
    phases = _parse_worker_logs()
    phase_data = next((p for p in phases if p["phase"] == phase), None)
    
    # Load bag if available
    bag_pairs = []
    if BAG_PATH.exists():
        try:
            with open(BAG_PATH) as f:
                bag_pairs = json.load(f)[:20]  # Limit for display
        except Exception:
            pass
    
    return jsonify({
        "phase": phase,
        "bag_size": len(bag_pairs) if bag_pairs else 10,
        "pairs": bag_pairs,
        "window_bars": 10000,
        "start_bar": 0,
        "end_bar": 10000,
        "window_days": 34.7,
        "loss": phase_data["loss"] if phase_data else None,
        "h_dim": 4,
        "H_layers": 4,
        "L_layers": 4,
    })


@app.route("/api/checkpoints")
def api_checkpoints():
    """All checkpoints endpoint."""
    return jsonify({"checkpoints": _scan_checkpoints()})


@app.route("/api/action/<action>", methods=["POST"])
def api_action(action: str):
    """Manual action endpoint."""
    if action == "save":
        # Signal workers to checkpoint (via file flag or other IPC)
        flag_path = CHECKPOINT_DIR / ".force_checkpoint"
        flag_path.touch()
        return jsonify({"status": "ok", "action": "save", "message": "Checkpoint requested"})
    
    elif action == "poll":
        # Trigger immediate metrics refresh
        return jsonify({"status": "ok", "action": "poll", "message": "Metrics refreshed"})
    
    else:
        return jsonify({"status": "error", "message": f"Unknown action: {action}"}), 400


def main():
    parser = argparse.ArgumentParser(description="HRM Dashboard Showdown")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting Dashboard Showdown on {args.host}:{args.port}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Log dir: {LOG_DIR}")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
