#!/usr/bin/env python3
"""
Autoresearch autonomous experiment loop.
Runs experiments with different hyperparameters, tracks val_bpb, and keeps improving.
"""

import os
import sys
import time
import subprocess
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "autoresearch"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
COMMIT_FILE = Path(__file__).parent / ".current_commit"

@dataclass
class Result:
    commit: str
    val_bpb: float
    memory_gb: float
    status: str
    depth: int
    aspect_ratio: int
    matrix_lr: float
    embedding_lr: float
    window_pattern: str
    description: str

def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, text=True, cwd=Path(__file__).parent
        )
        return result.stdout.strip()[:8]
    except:
        return "manual"

def save_commit():
    commit = get_git_commit()
    with open(COMMIT_FILE, "w") as f:
        f.write(commit)
    return commit

def read_results():
    results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 11:
                    results.append(Result(
                        commit=parts[0],
                        val_bpb=float(parts[1]),
                        memory_gb=float(parts[2]),
                        status=parts[3],
                        depth=int(parts[4]),
                        aspect_ratio=int(parts[5]),
                        matrix_lr=float(parts[6]),
                        embedding_lr=float(parts[7]),
                        window_pattern=parts[8],
                        description=parts[10] if len(parts) > 10 else ""
                    ))
    return results

def write_result(result: Result):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{result.commit}\t{result.val_bpb:.6f}\t{memory_gb:.2f}\t{result.status}\t"
                f"{result.depth}\t{result.aspect_ratio}\t{result.matrix_lr}\t{result.embedding_lr}\t"
                f"{result.window_pattern}\t{result.val_bpb:.6f}\t{result.description}\n")

def get_best_result():
    results = read_results()
    completed = [r for r in results if r.status == "ok"]
    if not completed:
        return None
    return min(completed, key=lambda r: r.val_bpb)

def modify_train_py(depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern):
    train_path = Path(__file__).parent / "train.py"
    content = train_path.read_text()
    
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("ASPECT_RATIO ="):
            new_lines.append(f"ASPECT_RATIO = {aspect_ratio}")
        elif line.startswith("DEPTH ="):
            new_lines.append(f"DEPTH = {depth}")
        elif line.startswith("WINDOW_PATTERN ="):
            new_lines.append(f'WINDOW_PATTERN = "{window_pattern}"')
        elif line.startswith("MATRIX_LR ="):
            new_lines.append(f"MATRIX_LR = {matrix_lr}")
        elif line.startswith("EMBEDDING_LR ="):
            new_lines.append(f"EMBEDDING_LR = {embedding_lr}")
        else:
            new_lines.append(line)
    
    train_path.write_text("\n".join(new_lines))

def run_experiment(depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description):
    print(f"\n{'='*60}")
    print(f"Experiment: {description}")
    print(f"depth={depth}, aspect_ratio={aspect_ratio}, matrix_lr={matrix_lr}, "
          f"embedding_lr={embedding_lr}, window={window_pattern}")
    print(f"{'='*60}")
    
    modify_train_py(depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern)
    commit = save_commit()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=Path(__file__).parent
        )
        
        output = result.stdout + result.stderr
        
        val_bpb = None
        memory_gb = None
        
        for line in output.split("\n"):
            if line.startswith("val_bpb:"):
                val_bpb = float(line.split(":")[1].strip())
            elif line.startswith("peak_vram_mb:"):
                memory_gb = float(line.split(":")[1].strip()) / 1024
        
        elapsed = time.time() - start_time
        
        if val_bpb is not None:
            status = "ok"
            print(f"Result: val_bpb={val_bpb:.4f}, memory={memory_gb:.1f}GB, time={elapsed:.0f}s")
            result_obj = Result(
                commit=commit,
                val_bpb=val_bpb,
                memory_gb=memory_gb or 0,
                status=status,
                depth=depth,
                aspect_ratio=aspect_ratio,
                matrix_lr=matrix_lr,
                embedding_lr=embedding_lr,
                window_pattern=window_pattern,
                description=description
            )
            write_result(result_obj)
            return result_obj
        else:
            print(f"Failed to parse val_bpb from output")
            print(output[-1000:])
            
    except subprocess.TimeoutExpired:
        print("Experiment timed out (>10min)")
    except Exception as e:
        print(f"Experiment failed: {e}")
    
    result_obj = Result(
        commit=commit,
        val_bpb=999,
        memory_gb=0,
        status="fail",
        depth=depth,
        aspect_ratio=aspect_ratio,
        matrix_lr=matrix_lr,
        embedding_lr=embedding_lr,
        window_pattern=window_pattern,
        description=description
    )
    write_result(result_obj)
    return None

def suggest_next_experiment(best: Result, iteration: int):
    depth = best.depth
    aspect_ratio = best.aspect_ratio
    matrix_lr = best.matrix_lr
    embedding_lr = best.embedding_lr
    window_pattern = best.window_pattern
    
    strategy = iteration % 5
    
    if strategy == 0:
        depth = depth + 1 if depth < 16 else depth - 1
        description = f"depth {depth} (from {best.depth})"
    elif strategy == 1:
        aspect_ratio = best.aspect_ratio + 8 if best.aspect_ratio < 96 else best.aspect_ratio - 8
        description = f"aspect_ratio {aspect_ratio} (from {best.aspect_ratio})"
    elif strategy == 2:
        matrix_lr = best.matrix_lr * 1.2 if best.matrix_lr < 0.1 else best.matrix_lr * 0.8
        description = f"matrix_lr {matrix_lr:.4f} (from {best.matrix_lr})"
    elif strategy == 3:
        embedding_lr = best.embedding_lr * 1.2 if best.embedding_lr < 1.0 else best.embedding_lr * 0.8
        description = f"embedding_lr {embedding_lr:.4f} (from {best.embedding_lr})"
    else:
        patterns = ["SSSS", "SSSL", "SSLL", "LLLL"]
        current_idx = patterns.index(window_pattern) if window_pattern in patterns else 0
        window_pattern = patterns[(current_idx + 1) % len(patterns)]
        description = f"window_pattern {window_pattern} (from {best.window_pattern})"
    
    return depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description

def main():
    print("Autoresearch Experiment Loop")
    print("=" * 40)
    
    print("\nRunning baseline experiment...")
    run_experiment(
        depth=8,
        aspect_ratio=64,
        matrix_lr=0.04,
        embedding_lr=0.6,
        window_pattern="SSSL",
        description="baseline"
    )
    
    best = get_best_result()
    if not best:
        print("Baseline failed, trying simpler config...")
        run_experiment(
            depth=6,
            aspect_ratio=48,
            matrix_lr=0.03,
            embedding_lr=0.4,
            window_pattern="SSSS",
            description="simple baseline"
        )
        best = get_best_result()
    
    if not best:
        print("All baselines failed. Check errors.")
        return
    
    print(f"\nBest so far: val_bpb={best.val_bpb:.4f}")
    
    iteration = 0
    max_iterations = 50
    
    while iteration < max_iterations:
        iteration += 1
        depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description = suggest_next_experiment(best, iteration)
        
        result = run_experiment(depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description)
        
        if result and result.val_bpb < best.val_bpb:
            improvement = best.val_bpb - result.val_bpb
            print(f"*** NEW BEST! improved by {improvement:.6f} ***")
            best = result
        else:
            print(f"No improvement (best: {best.val_bpb:.4f})")
        
        print(f"\nIteration {iteration}/{max_iterations}. Best: val_bpb={best.val_bpb:.4f}")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    results = read_results()
    for r in sorted(results, key=lambda x: x.val_bpb)[:10]:
        print(f"{r.val_bpb:.4f} | {r.description} | depth={r.depth} ar={r.aspect_ratio}")

if __name__ == "__main__":
    main()
