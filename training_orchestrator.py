#!/usr/bin/env python3
"""
Training Orchestrator - Continuous HRM training with smart scheduling.

Manages two distinct training tracks:
1. **PRETRAIN**: Stochastic bags on bulk Binance data (many pairs, long history)
2. **DAYTRADE**: Fixed bag on Coinbase live data (recent window, paper trade signals)

Also manages service lifecycle via Docker Compose.

Schedule:
- PRETRAIN: Runs continuously, rotates every 6h if stagnant
- DAYTRADE: Runs continuously, fine-tunes every hour on recent window
- PRUNE: Weekly cleanup of defective checkpoints

Usage:
  python3 training_orchestrator.py --mode pretrain --continuous
  python3 training_orchestrator.py --mode daytrade --continuous
  python3 training_orchestrator.py --services-restart training-pretrain
  crontab -e  # Add the cronjobs from below
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import schedule

# Configuration
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "./checkpoints"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
DOCKER_COMPOSE_FILE = Path(os.getenv("DOCKER_COMPOSE_FILE", "./docker-compose.yml"))

# Training config
PRETRAIN_SAVE_EVERY = int(os.getenv("PRETRAIN_SAVE_EVERY", "100"))
DAYTRADE_RETRAIN_MINUTES = int(os.getenv("DAYTRADE_RETRAIN_MINUTES", "60"))
STAGNANT_THRESHOLD = float(os.getenv("STAGNANT_THRESHOLD", "0.001"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'orchestrator.log') if LOG_DIR.exists() else logging.StreamHandler()
    ]
)
logger = logging.getLogger('training_orchestrator')


class TrainingJob:
    """Represents a training job configuration."""
    
    MODES = {
        "pretrain": {
            "service": "training-pretrain",
            "mode": "pretrain",
            "bag_type": "stochastic",
            "data_source": "binance",
            "checkpoint_name": "model_weights_pretrained.pt",
            "description": "Stochastic bags on bulk Binance historical data",
        },
        "daytrade": {
            "service": "training-finetune",
            "mode": "finetune",
            "bag_type": "fixed",
            "data_source": "coinbase",
            "checkpoint_name": "model_weights_daytrade.pt",
            "description": "Fixed bag on Coinbase live data with paper trading",
        },
    }
    
    def __init__(self, mode: str):
        self.mode = mode
        self.config = self.MODES.get(mode)
        if not self.config:
            raise ValueError(f"Unknown mode: {mode}. Valid: {list(self.MODES.keys())}")
    
    @property
    def service_name(self) -> str:
        return self.config["service"]
    
    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / self.config["checkpoint_name"]
    
    def get_stats(self) -> Dict:
        """Get training stats for this job."""
        stats = {
            "mode": self.mode,
            "service": self.service_name,
            "checkpoint_exists": self.checkpoint_path.exists(),
        }
        
        if self.checkpoint_path.exists():
            stat = self.checkpoint_path.stat()
            stats["checkpoint_size_mb"] = round(stat.st_size / 1024 / 1024, 2)
            stats["checkpoint_age_hours"] = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600
        
        return stats


def docker_compose_cmd(args: List[str]) -> Tuple[int, str, str]:
    """Run docker compose command."""
    cmd = ['docker', 'compose']
    if DOCKER_COMPOSE_FILE.exists():
        cmd = ['docker', 'compose', '-f', str(DOCKER_COMPOSE_FILE)]
    cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(DOCKER_COMPOSE_FILE.parent) if DOCKER_COMPOSE_FILE.exists() else "."
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def get_service_status(service_name: str) -> Optional[Dict]:
    """Get status of a specific service."""
    code, stdout, _ = docker_compose_cmd(['ps', service_name, '--format', 'json'])
    if code != 0 or not stdout.strip():
        return None
    
    try:
        data = json.loads(stdout.strip().split('\n')[0])
        return {
            "name": data.get("Name"),
            "state": data.get("State", "unknown").lower(),
            "status": data.get("Status", "unknown"),
            "health": data.get("Health"),
        }
    except json.JSONDecodeError:
        return None


def start_service(service_name: str) -> bool:
    """Start a Docker service."""
    logger.info(f"Starting service: {service_name}")
    code, stdout, stderr = docker_compose_cmd(['up', '-d', service_name])
    if code != 0:
        logger.error(f"Failed to start {service_name}: {stderr}")
        return False
    logger.info(f"Started {service_name}")
    return True


def stop_service(service_name: str) -> bool:
    """Stop a Docker service."""
    logger.info(f"Stopping service: {service_name}")
    code, stdout, stderr = docker_compose_cmd(['stop', service_name])
    if code != 0:
        logger.error(f"Failed to stop {service_name}: {stderr}")
        return False
    logger.info(f"Stopped {service_name}")
    return True


def restart_service(service_name: str) -> bool:
    """Restart a Docker service."""
    logger.info(f"Restarting service: {service_name}")
    code, stdout, stderr = docker_compose_cmd(['restart', service_name])
    if code != 0:
        logger.error(f"Failed to restart {service_name}: {stderr}")
        return False
    logger.info(f"Restarted {service_name}")
    return True


def get_training_loss_history(mode: str, n_lines: int = 50) -> List[Dict]:
    """Extract loss history from training logs."""
    import re
    
    job = TrainingJob(mode)
    log_file = LOG_DIR / f"{job.service_name}.log"
    
    if not log_file.exists():
        return []
    
    phases = []
    try:
        with open(log_file) as f:
            lines = f.readlines()
        
        for line in lines[-n_lines:]:
            # Phase 123: loss=0.456789, updates=100
            match = re.search(r'Phase\s+(\d+).*loss=([\d.nan]+)', line, re.IGNORECASE)
            if match:
                phase = int(match.group(1))
                loss_str = match.group(2)
                try:
                    loss = float(loss_str) if loss_str != 'nan' else float('nan')
                    phases.append({"phase": phase, "loss": loss, "line": line.strip()[:100]})
                except ValueError:
                    pass
    except Exception as e:
        logger.warning(f"Failed to read log {log_file}: {e}")
    
    return phases


def is_stagnant(phases: List[Dict], window: int = 10) -> Tuple[bool, float]:
    """Check if training is stagnant (no improvement)."""
    if len(phases) < window:
        return False, 0.0
    
    recent_losses = [p["loss"] for p in phases[-window:] if p["loss"] == p["loss"]]  # filter NaN
    if len(recent_losses) < window // 2:
        return True, float('inf')  # Too many NaN
    
    if len(recent_losses) < 2:
        return False, 0.0
    
    variance = max(recent_losses) - min(recent_losses)
    return variance < STAGNANT_THRESHOLD, variance


def run_pretrain_cycle():
    """Run one pretraining cycle with stagnation detection."""
    job = TrainingJob("pretrain")
    logger.info(f"Pretrain cycle starting for {job.service_name}")
    
    status = get_service_status(job.service_name)
    if not status or status.get("state") != "running":
        logger.warning(f"{job.service_name} not running, starting...")
        start_service(job.service_name)
        return
    
    # Check for stagnation
    phases = get_training_loss_history("pretrain", n_lines=200)
    is_stuck, variance = is_stagnant(phases, window=20)
    
    if is_stuck:
        logger.warning(f"PRETRAIN STAGNANT (variance: {variance:.6f}), rotating...")
        
        # Save current checkpoint with stagnation marker
        if job.checkpoint_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stale_path = CHECKPOINT_DIR / f"model_weights_pretrained_stale_{ts}.pt"
            import shutil
            shutil.copy(job.checkpoint_path, stale_path)
            logger.info(f"Backed up stale checkpoint to {stale_path}")
        
        # Restart fresh
        restart_service(job.service_name)
    else:
        logger.debug(f"Pretrain healthy (variance: {variance:.6f})")


def run_daytrade_cycle():
    """Run one daytrade retraining cycle."""
    job = TrainingJob("daytrade")
    logger.info(f"Daytrade retrain cycle for {job.service_name}")
    
    status = get_service_status(job.service_name)
    if not status or status.get("state") != "running":
        logger.warning(f"{job.service_name} not running, starting...")
        start_service(job.service_name)
        return
    
    # For daytrade, just check it's running and has recent activity
    phases = get_training_loss_history("daytrade", n_lines=10)
    
    if not phases:
        logger.warning("Daytrade has no recent activity, checking service...")
        restart_service(job.service_name)
    else:
        last_phase = phases[-1]
        logger.info(f"Daytrade active: Phase {last_phase['phase']}, loss={last_phase['loss']:.6f}")


def run_health_check_all():
    """Health check all training-related services."""
    services = ["training-pretrain", "training-finetune", "data-ingestion", "paper-trader"]
    
    for svc in services:
        status = get_service_status(svc)
        if not status or status.get("state") != "running":
            logger.warning(f"Service {svc} not running, restarting...")
            start_service(svc)
        else:
            logger.debug(f"Service {svc} healthy: {status.get('status')}")


def print_cron_setup():
    """Print recommended cron configuration."""
    print("""
# ============================================
# HRM Training Orchestrator Cronjobs
# Add to crontab with: crontab -e
# ============================================

# Every 5 minutes: Health check all services, restart if down
*/5 * * * * cd /app && /usr/bin/python3 /app/training_orchestrator.py --services-health-check >> /app/logs/orchestrator_cron.log 2>&1

# Every 30 minutes: Check pretraining progress, rotate if stagnant
*/30 * * * * cd /app && /usr/bin/python3 /app/training_orchestrator.py --check-pretrain >> /app/logs/orchestrator_cron.log 2>&1

# Every hour: Ensure daytrade fine-tuning is running
0 * * * * cd /app && /usr/bin/python3 /app/training_orchestrator.py --check-daytrade >> /app/logs/orchestrator_cron.log 2>&1

# Daily at 3 AM: Full restart of training pipeline for clean slate
0 3 * * * cd /app && /usr/bin/python3 /app/training_orchestrator.py --restart-all >> /app/logs/orchestrator_cron.log 2>&1

# Daily at 4 AM: Prune defective/old checkpoints (keep last 10 per mode)
0 4 * * * cd /app && /usr/bin/python3 /app/training_orchestrator.py --prune-checkpoints >> /app/logs/orchestrator_cron.log 2>&1

# ============================================
# Alternative: Run orchestrator as daemon with schedule library
# /usr/bin/python3 /app/training_orchestrator.py --daemon
# ============================================
""")


def prune_old_checkpoints(keep_last: int = 10) -> int:
    """Remove old checkpoints, keep last N per mode."""
    removed = 0
    
    for pattern, desc in [
        ("model_weights_pretrained_*.pt", "pretrained"),
        ("model_weights_daytrade_*.pt", "daytrade"),
    ]:
        checkpoints = sorted(
            CHECKPOINT_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_cp in checkpoints[keep_last:]:
            try:
                old_cp.unlink()
                logger.info(f"Pruned old checkpoint: {old_cp.name}")
                removed += 1
            except Exception as e:
                logger.error(f"Failed to prune {old_cp}: {e}")
    
    return removed


def run_daemon():
    """Run as continuous daemon with internal scheduling."""
    logger.info("=" * 60)
    logger.info("HRM Training Orchestrator Daemon Starting")
    logger.info("=" * 60)
    
    # Schedule tasks
    schedule.every(5).minutes.do(run_health_check_all)
    schedule.every(30).minutes.do(run_pretrain_cycle)
    schedule.every().hour.do(run_daytrade_cycle)
    schedule.every().day.at("03:00").do(lambda: restart_service("training-pretrain"))
    schedule.every().day.at("03:30").do(lambda: restart_service("training-finetune"))
    schedule.every().day.at("04:00").do(lambda: prune_old_checkpoints(keep_last=10))
    
    logger.info("Scheduled tasks:")
    for job in schedule.get_jobs():
        logger.info(f"  {job}")
    
    # Main loop
    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="HRM Training Orchestrator")
    
    # Action modes
    parser.add_argument("--services-health-check", action="store_true",
                       help="Health check all services, restart if unhealthy")
    parser.add_argument("--check-pretrain", action="store_true",
                       help="Check pretraining, rotate if stagnant")
    parser.add_argument("--check-daytrade", action="store_true",
                       help="Check daytrade fine-tuning")
    parser.add_argument("--restart-all", action="store_true",
                       help="Restart all training services")
    parser.add_argument("--prune-checkpoints", action="store_true",
                       help="Remove old checkpoint files (keep last 10)")
    parser.add_argument("--print-cron", action="store_true",
                       help="Print recommended cron configuration")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as continuous daemon with internal scheduling")
    parser.add_argument("--status", action="store_true",
                       help="Show current status of all training jobs")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.print_cron:
        print_cron_setup()
        return 0
    
    if args.daemon:
        run_daemon()
        return 0
    
    if args.status:
        for mode in ["pretrain", "daytrade"]:
            job = TrainingJob(mode)
            stats = job.get_stats()
            print(f"\n{mode.upper()}:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        return 0
    
    if args.services_health_check:
        run_health_check_all()
        return 0
    
    if args.check_pretrain:
        run_pretrain_cycle()
        return 0
    
    if args.check_daytrade:
        run_daytrade_cycle()
        return 0
    
    if args.restart_all:
        for svc in ["training-pretrain", "training-finetune", "paper-trader", "data-ingestion"]:
            restart_service(svc)
        return 0
    
    if args.prune_checkpoints:
        removed = prune_old_checkpoints(keep_last=10)
        print(f"Pruned {removed} old checkpoints")
        return 0
    
    # Default: show help
    parser.print_help()
    print("\nTip: Use --print-cron to see recommended cron setup")
    return 1


if __name__ == "__main__":
    sys.exit(main())
