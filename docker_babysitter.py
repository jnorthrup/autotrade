#!/usr/bin/env python3
"""
Docker Babysitter - Cron-based service health monitor and automatic repair.

Monitors all Docker services, detects failures, restarts bad ones,
prunes defective checkpoints (NaN, stale, unimproving), and
manages training job lifecycle.

Schedule as a cronjob:
  */5 * * * * /usr/bin/python3 /app/docker_babysitter.py --check-services
  0 */6 * * * /usr/bin/python3 /app/docker_babysitter.py --prune-defective
  0 2 * * * /usr/bin/python3 /app/docker_babysitter.py --rotate-training

Usage:
  python3 docker_babysitter.py --check-services    # Quick health check
  python3 docker_babysitter.py --prune-defective   # Remove bad checkpoints
  python3 docker_babysitter.py --rotate-training   # Start fresh training if stuck
  python3 docker_babysitter.py --run-continuous    # Run as daemon with schedule
"""

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

import schedule

# Configuration
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "./checkpoints"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
DOCKER_COMPOSE_FILE = Path(os.getenv("DOCKER_COMPOSE_FILE", "./docker-compose.yml"))

# Defective checkpoint criteria
MAX_CHECKPOINT_AGE_HOURS = 48  # Consider stale after 2 days without improvement
MIN_IMPROVEMENT_THRESHOLD = 0.001  # Loss must improve by at least 0.1%
STUCK_PHASE_COUNT = 10  # Consider stuck if loss same for N phases

# Training quality thresholds
NAN_LOSS_THRESHOLD = 1  # Phases with NaN before marked defective
EXPLODING_LOSS_THRESHOLD = 10.0  # Loss above this is broken

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'babysitter.log') if LOG_DIR.exists() else logging.StreamHandler()
    ]
)
logger = logging.getLogger('docker_babysitter')


class ServiceInfo:
    """Represents a Docker service state."""
    def __init__(self, name: str, status: str, health: str = None, 
                 uptime: str = None, restarts: int = 0):
        self.name = name
        self.status = status
        self.health = health
        self.uptime = uptime
        self.restarts = restarts
        self.is_healthy = status == 'running' and (health is None or health == 'healthy')
    
    def __repr__(self):
        health_str = f" health={self.health}" if self.health else ""
        return f"<{self.name}: {self.status}{health_str} restarts={self.restarts}>"


def docker_compose_cmd(args: List[str]) -> Tuple[int, str, str]:
    """Run docker compose command, return (exit_code, stdout, stderr)."""
    cmd = ['docker', 'compose'] + args
    if DOCKER_COMPOSE_FILE.exists():
        cmd = ['docker', 'compose', '-f', str(DOCKER_COMPOSE_FILE)] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(DOCKER_COMPOSE_FILE.parent) if DOCKER_COMPOSE_FILE.exists() else "."
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def get_service_status() -> List[ServiceInfo]:
    """Get status of all Docker Compose services."""
    services = []
    
    # Get service list and status
    code, stdout, stderr = docker_compose_cmd(['ps', '--format', 'json'])
    if code != 0:
        logger.error(f"Failed to get service status: {stderr}")
        return services
    
    try:
        for line in stdout.strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            
            name = data.get('Name', 'unknown')
            state = data.get('State', 'unknown')
            status = data.get('Status', 'unknown')
            health = data.get('Health', None)
            
            # Parse uptime/restarts from status
            uptime = None
            restarts = 0
            
            # Extract restart count if present
            restart_match = re.search(r'(\d+) \(.*\)', status)
            if restart_match:
                restarts = int(restart_match.group(1))
            
            services.append(ServiceInfo(
                name=name,
                status=state.lower(),
                health=health,
                uptime=uptime,
                restarts=restarts
            ))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse docker output: {e}")
    
    return services


def restart_service(service_name: str) -> bool:
    """Restart a specific service."""
    logger.warning(f"Restarting service: {service_name}")
    
    code, stdout, stderr = docker_compose_cmd(['restart', service_name])
    if code != 0:
        logger.error(f"Failed to restart {service_name}: {stderr}")
        return False
    
    logger.info(f"Successfully restarted {service_name}")
    return True


def restart_unhealthy_services() -> int:
    """Check and restart any unhealthy services. Returns count restarted."""
    services = get_service_status()
    restarted = 0
    
    for svc in services:
        if not svc.is_healthy:
            logger.warning(f"Unhealthy service detected: {svc}")
            if restart_service(svc.name):
                restarted += 1
    
    return restarted


def analyze_checkpoint_quality(checkpoint_path: Path) -> Dict:
    """Analyze a checkpoint file for quality metrics."""
    import torch
    
    result = {
        "path": str(checkpoint_path),
        "size_mb": round(checkpoint_path.stat().st_size / 1024 / 1024, 2),
        "timestamp": datetime.fromtimestamp(checkpoint_path.stat().st_mtime),
        "has_nan": False,
        "exploded": False,
        "dimensions": None,
        "age_hours": None,
    }
    
    result["age_hours"] = (datetime.now() - result["timestamp"]).total_seconds() / 3600
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check for NaN values
        for key, tensor in checkpoint.items():
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any():
                    result["has_nan"] = True
                    break
                # Check for exploding values
                if tensor.abs().max() > EXPLODING_LOSS_THRESHOLD * 100:
                    result["exploded"] = True
        
        # Extract dimensions if present
        if 'h_dim' in checkpoint and 'H_layers' in checkpoint:
            result["dimensions"] = {
                "h_dim": checkpoint['h_dim'],
                "H_layers": checkpoint['H_layers'],
                "L_layers": checkpoint['L_layers'],
            }
    except Exception as e:
        logger.warning(f"Failed to analyze {checkpoint_path}: {e}")
        result["corrupted"] = True
    
    return result


def get_training_history_from_logs() -> List[Dict]:
    """Parse training logs to extract loss history."""
    phases = []
    
    for log_file in LOG_DIR.glob("*training*.log"):
        try:
            with open(log_file) as f:
                for line in f:
                    # Match: "Phase N: loss=X.XXXXXX"
                    match = re.search(r'Phase\s+(\d+).*loss=([\d.]+|nan)', line, re.IGNORECASE)
                    if match:
                        phase = int(match.group(1))
                        loss_str = match.group(2)
                        loss = float(loss_str) if loss_str != 'nan' else float('nan')
                        
                        phases.append({
                            "phase": phase,
                            "loss": loss,
                            "timestamp": line[:19] if len(line) > 19 else None,
                        })
        except Exception as e:
            logger.warning(f"Failed to parse {log_file}: {e}")
    
    return sorted(phases, key=lambda x: x["phase"])


def is_training_defective(phases: List[Dict], checkpoint_age_hours: float) -> Tuple[bool, str]:
    """Determine if training is defective based on metrics."""
    
    if not phases:
        return checkpoint_age_hours > MAX_CHECKPOINT_AGE_HOURS, "No training phases found"
    
    recent = phases[-20:]  # Look at last 20 phases
    
    # Check for NaN losses
    nan_count = sum(1 for p in recent if p.get('loss') != p.get('loss'))  # NaN != NaN
    if nan_count >= NAN_LOSS_THRESHOLD:
        return True, f"Too many NaN losses: {nan_count}"
    
    # Check for exploding losses
    for p in recent:
        if p.get('loss', 0) > EXPLODING_LOSS_THRESHOLD:
            return True, f"Exploding loss: {p.get('loss')}"
    
    # Check for stagnation
    if len(recent) >= STUCK_PHASE_COUNT:
        last_losses = [p.get('loss') for p in recent[-STUCK_PHASE_COUNT:] if p.get('loss') == p.get('loss')]
        if len(last_losses) >= STUCK_PHASE_COUNT:
            loss_variance = max(last_losses) - min(last_losses)
            if loss_variance < MIN_IMPROVEMENT_THRESHOLD:
                return True, f"Stagnant loss (variance: {loss_variance:.6f})"
    
    # Check staleness
    if checkpoint_age_hours > MAX_CHECKPOINT_AGE_HOURS:
        return True, f"Stale checkpoint: {checkpoint_age_hours:.1f} hours old"
    
    return False, "Training appears healthy"


def prune_defective_checkpoints(dry_run: bool = False) -> int:
    """Remove defective checkpoints. Returns count removed."""
    removed = 0
    
    if not CHECKPOINT_DIR.exists():
        logger.warning(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return 0
    
    phases = get_training_history_from_logs()
    
    for checkpoint_file in CHECKPOINT_DIR.glob("*.pt"):
        analysis = analyze_checkpoint_quality(checkpoint_file)
        
        is_defective, reason = is_training_defective(phases, analysis.get('age_hours', 0))
        
        # Also flag corrupted files
        if analysis.get('corrupted') or analysis.get('has_nan') or analysis.get('exploded'):
            is_defective = True
            reason = "Corrupted/NaN/Exploded"
        
        if is_defective:
            logger.warning(f"Defective checkpoint: {checkpoint_file.name}")
            logger.warning(f"  Reason: {reason}")
            logger.warning(f"  Analysis: {analysis}")
            
            if not dry_run:
                try:
                    backup_path = checkpoint_file.with_suffix('.pt.defective')
                    shutil.move(checkpoint_file, backup_path)
                    logger.info(f"Moved to backup: {backup_path}")
                    removed += 1
                except Exception as e:
                    logger.error(f"Failed to backup {checkpoint_file}: {e}")
        else:
            logger.debug(f"Healthy checkpoint: {checkpoint_file.name}")
    
    return removed


def rotate_training_job(job_type: str = "pretrain") -> bool:
    """Stop current training and start fresh."""
    logger.info(f"Rotating {job_type} training job")
    
    service_map = {
        "pretrain": "training-pretrain",
        "finetune": "training-finetune",
    }
    
    service_name = service_map.get(job_type)
    if not service_name:
        logger.error(f"Unknown job type: {job_type}")
        return False
    
    # Stop the service
    code, _, stderr = docker_compose_cmd(['stop', service_name])
    if code != 0:
        logger.error(f"Failed to stop {service_name}: {stderr}")
        return False
    
    # Rotate logs
    log_file = LOG_DIR / f"{service_name}.log"
    if log_file.exists():
        backup = log_file.with_suffix(f'.log.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        shutil.move(log_file, backup)
        logger.info(f"Rotated log to {backup}")
    
    # Start fresh
    code, _, stderr = docker_compose_cmd(['up', '-d', service_name])
    if code != 0:
        logger.error(f"Failed to start {service_name}: {stderr}")
        return False
    
    logger.info(f"Successfully rotated {service_name}")
    return True


def check_all_services() -> Dict:
    """Comprehensive service health check."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "defective_training_detected": False,
        "actions_taken": [],
    }
    
    # Check service health
    services = get_service_status()
    for svc in services:
        report["services"][svc.name] = {
            "status": svc.status,
            "health": svc.health,
            "restarts": svc.restarts,
            "is_healthy": svc.is_healthy,
        }
    
    # Restart unhealthy
    restarted = restart_unhealthy_services()
    if restarted > 0:
        report["actions_taken"].append(f"restarted {restarted} unhealthy services")
    
    # Check training quality
    phases = get_training_history_from_logs()
    defective, reason = is_training_defective(phases, MAX_CHECKPOINT_AGE_HOURS)
    report["defective_training_detected"] = defective
    report["defective_reason"] = reason
    
    return report


def run_scheduler():
    """Run the scheduled babysitter tasks."""
    logger.info("Docker Babysitter scheduler starting...")
    
    # Every 5 minutes: check service health
    schedule.every(5).minutes.do(lambda: check_all_services())
    
    # Every 6 hours: prune defective checkpoints
    schedule.every(6).hours.do(lambda: prune_defective_checkpoints())
    
    # Daily at 2 AM: rotate pretrain if stagnating
    schedule.every().day.at("02:00").do(lambda: rotate_training_job("pretrain"))
    
    # Daily at 3 AM: rotate finetune
    schedule.every().day.at("03:00").do(lambda: rotate_training_job("finetune"))
    
    logger.info("Scheduler running. Press Ctrl+C to stop.")
    
    while True:
        schedule.run_pending()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Docker Babysitter")
    parser.add_argument("--check-services", action="store_true", 
                       help="Quick service health check")
    parser.add_argument("--prune-defective", action="store_true",
                       help="Remove defective checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without doing it")
    parser.add_argument("--rotate-training",
                       help="Rotate training job (pretrain/finetune)")
    parser.add_argument("--run-continuous", action="store_true",
                       help="Run scheduler continuously")
    
    args = parser.parse_args()
    
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.check_services:
        report = check_all_services()
        print(json.dumps(report, indent=2, default=str))
        return 0 if not report.get("defective_training_detected") else 1
    
    elif args.prune_defective:
        removed = prune_defective_checkpoints(dry_run=args.dry_run)
        print(f"{'Would remove' if args.dry_run else 'Removed'} {removed} defective checkpoints")
        return 0
    
    elif args.rotate_training:
        success = rotate_training_job(args.rotate_training)
        return 0 if success else 1
    
    elif args.run_continuous:
        run_scheduler()
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
