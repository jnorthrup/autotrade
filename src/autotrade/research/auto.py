#!/usr/bin/env python3
"""
Physical Autoresearch Loop (MLX + ANE).
Optimizes for 5-minute interval entropy.
"""

import os
import sys
import time
import subprocess
import random
import csv
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure results file is absolute and in the correct directory
RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(RESEARCH_DIR, "results_physical.tsv")

@dataclass
class Result:
    val_acc: float
    description: str
    curvature: float
    power: float
    h_cycles: int
    l_cycles: int
    status: str = "ok"

def read_results():
    results = []
    if not os.path.exists(RESULTS_FILE): return results
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(Result(
                val_acc=float(row["val_acc"]),
                description=row["description"],
                curvature=float(row["curvature"]),
                power=float(row["power"]),
                h_cycles=int(row["h_cycles"]),
                l_cycles=int(row["l_cycles"]),
                status=row.get("status", "ok")
            ))
    return results

def run_experiment(curvature, power, h_cycles, l_cycles, description):
    print(f"\n--- Physical Experiment (5m): {description} ---")
    
    train_file = os.path.join(os.path.dirname(__file__), "train.py")
    with open(train_file, "r") as f: lines = f.readlines()
    with open(train_file, "w") as f:
        for line in lines:
            if line.startswith("CURVATURE = "): f.write(f"CURVATURE = {curvature}\n")
            elif line.startswith("HYPERBOLIC_POWER = "): f.write(f"HYPERBOLIC_POWER = {power}\n")
            elif line.startswith("H_CYCLES = "): f.write(f"H_CYCLES = {h_cycles}\n")
            elif line.startswith("L_CYCLES = "): f.write(f"L_CYCLES = {l_cycles}\n")
            else: f.write(line)
            
    # Run from project root as a module to fix relative imports
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    print(f"Executing experiment in root: {root_dir}")
    cmd = ["uv", "run", "-m", "src.autotrade.research.train"]
    try:
        # Increase timeout or check for immediate failure
        print("Starting training...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=root_dir)
        print("Training finished successfully.")
        val_acc = None
        for line in result.stdout.splitlines():
            if "Final val_acc:" in line:
                val_acc = float(line.split(":")[1].strip())
                break
        if val_acc is None: 
            print("Failed to parse val_acc. stdout:")
            print(result.stdout)
            return None
        
        # Log to file manually
        with open(RESULTS_FILE, "a") as f:
            if os.path.getsize(RESULTS_FILE) == 0:
                f.write("val_acc\tdescription\tcurvature\tpower\th_cycles\tl_cycles\tstatus\n")
            f.write(f"{val_acc}\t{description}\t{curvature}\t{power}\t{h_cycles}\t{l_cycles}\tok\n")
        return Result(val_acc, description, curvature, power, h_cycles, l_cycles)
    except Exception as e:
        print(f"Experiment failed: {e}")
        if hasattr(e, 'stdout') and e.stdout: print(e.stdout)
        if hasattr(e, 'stderr') and e.stderr: print(e.stderr)
        return None

def suggest_next_experiment(results: list[Result], iteration: int):
    if len(results) < 5:
        # Initial Exploration
        return random.uniform(0.5, 4.0), random.uniform(1.0, 3.0), random.randint(1, 4), random.randint(1, 6), f"exploration {iteration}"

    # Meta-Optimizer
    X = torch.tensor([[r.curvature, r.power, r.h_cycles, r.l_cycles] for r in results if r.status == "ok"], dtype=torch.float32)
    y = torch.tensor([[r.val_acc] for r in results if r.status == "ok"], dtype=torch.float32)
    X_m, X_s = X.mean(0), X.std(0) + 1e-6
    y_m, y_s = y.mean(), y.std() + 1e-6
    
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(500):
        opt.zero_grad(); nn.MSELoss()(model((X-X_m)/X_s), (y-y_m)/y_s).backward(); opt.step()
    
    best_c, best_v = None, -float('inf')
    for _ in range(200):
        c = [random.uniform(0.5, 4.0), random.uniform(1.0, 3.5), random.randint(1, 4), random.randint(1, 6)]
        with torch.no_grad():
            v = model((torch.tensor([c], dtype=torch.float32)-X_m)/X_s).item() * y_s.item() + y_m.item()
        if v > best_v: best_v, best_c = v, c
            
    curv, pwr, h, l = best_c
    return curv, pwr, h, l, f"model-suggested (pred_acc={best_v:.4f})"

def main():
    print("Physical 5m HRM-ANE Experiment Loop")
    iteration = 0
    while iteration < 50:
        iteration += 1
        results = read_results()
        curv, pwr, h, l, desc = suggest_next_experiment(results, iteration)
        run_experiment(curv, pwr, h, l, desc)
        
        # Re-read results to get latest best
        updated_results = read_results()
        best = max(updated_results, key=lambda x: x.val_acc) if updated_results else None
        best_acc = best.val_acc if best else 0.0
        print(f"\nIteration {iteration}. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
