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
import csv
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

RESULTS_FILE = "results.tsv"

@dataclass
class Result:
    val_bpb: float
    description: str
    depth: int
    aspect_ratio: int
    matrix_lr: float
    embedding_lr: float
    window_pattern: str
    status: str = "ok"

def read_results():
    results = []
    if not os.path.exists(RESULTS_FILE):
        return results
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(Result(
                val_bpb=float(row["val_bpb"]),
                description=row["description"],
                depth=int(row["depth"]),
                aspect_ratio=int(row["aspect_ratio"]),
                matrix_lr=float(row["matrix_lr"]),
                embedding_lr=float(row["embedding_lr"]),
                window_pattern=row["window_pattern"],
                status=row.get("status", "ok")
            ))
    return results

def write_result(r: Result):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=[
            "val_bpb", "description", "depth", "aspect_ratio", "matrix_lr", "embedding_lr", "window_pattern", "status"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "val_bpb": r.val_bpb,
            "description": r.description,
            "depth": r.depth,
            "aspect_ratio": r.aspect_ratio,
            "matrix_lr": r.matrix_lr,
            "embedding_lr": r.embedding_lr,
            "window_pattern": r.window_pattern,
            "status": r.status
        })

def get_best_result():
    results = [r for r in read_results() if r.status == "ok"]
    if not results:
        return None
    return min(results, key=lambda x: x.val_bpb)

def run_experiment(depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description):
    print(f"\n--- Experiment: {description} ---")
    print(f"Params: depth={depth} ar={aspect_ratio} mlr={matrix_lr:.4f} elr={embedding_lr:.4f} win={window_pattern}")
    
    # Update train.py
    with open("train.py", "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith("DEPTH = "):
            new_lines.append(f"DEPTH = {depth}\n")
        elif line.startswith("ASPECT_RATIO = "):
            new_lines.append(f"ASPECT_RATIO = {aspect_ratio}\n")
        elif line.startswith("MATRIX_LR = "):
            new_lines.append(f"MATRIX_LR = {matrix_lr}\n")
        elif line.startswith("EMBEDDING_LR = "):
            new_lines.append(f"EMBEDDING_LR = {embedding_lr}\n")
        elif line.startswith("WINDOW_PATTERN = "):
            new_lines.append(f"WINDOW_PATTERN = \"{window_pattern}\"\n")
        else:
            new_lines.append(line)
            
    with open("train.py", "w") as f:
        f.writelines(new_lines)
        
    # Run training
    cmd = ["uv", "run", "train.py"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse output for val_bpb
        val_bpb = None
        for line in result.stdout.splitlines():
            if line.startswith("val_bpb:"):
                val_bpb = float(line.split(":")[1].strip())
                break
        
        if val_bpb is None:
            print("Failed to find val_bpb in output")
            return None
            
        res = Result(val_bpb, description, depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern)
        write_result(res)
        return res
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with exit code {e.returncode}")
        print(e.stderr)
        res = Result(999.0, description, depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, status="fail")
        write_result(res)
        return None

class ParamPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def window_to_idx(p):
    patterns = ["SSSS", "SSSL", "SSLL", "LLLL"]
    return patterns.index(p) if p in patterns else 0

def suggest_next_experiment(results: list[Result], iteration: int):
    if len(results) < 5:
        depth = random.randint(4, 16)
        aspect_ratio = random.choice([32, 48, 64, 80, 96])
        matrix_lr = random.uniform(0.01, 0.1)
        embedding_lr = random.uniform(0.1, 1.0)
        window_pattern = random.choice(["SSSS", "SSSL", "SSLL", "LLLL"])
        description = f"exploration {iteration}"
        return depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description

    X, y = [], []
    for r in results:
        if r.status == "ok":
            X.append([r.depth, r.aspect_ratio, r.matrix_lr, r.embedding_lr, window_to_idx(r.window_pattern)])
            y.append([r.val_bpb])
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    X_mean, X_std = X.mean(0), X.std(0) + 1e-6
    y_mean, y_std = y.mean(), y.std() + 1e-6
    
    model = ParamPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(500):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model((X - X_mean) / X_std), (y - y_mean) / y_std)
        loss.backward()
        optimizer.step()
    
    best_candidate = None
    best_pred_val = float('inf')
    for _ in range(100):
        c = [random.randint(4, 16), random.choice([32, 48, 64, 80, 96]), random.uniform(0.01, 0.1), random.uniform(0.1, 1.0), random.randint(0, 3)]
        with torch.no_grad():
            pred = model((torch.tensor([c], dtype=torch.float32) - X_mean) / X_std).item() * y_std.item() + y_mean.item()
        if pred < best_pred_val:
            best_pred_val, best_candidate = pred, c
            
    depth, aspect_ratio, matrix_lr, embedding_lr, win_idx = best_candidate
    window_pattern = ["SSSS", "SSSL", "SSLL", "LLLL"][win_idx]
    description = f"model-suggested (pred_bpb={best_pred_val:.4f})"
    return depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description

def main():
    print("Autoresearch Experiment Loop")
    print("=" * 40)
    
    iteration = 0
    max_iterations = 50
    while iteration < max_iterations:
        iteration += 1
        results = read_results()
        depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description = suggest_next_experiment(results, iteration)
        run_experiment(depth, aspect_ratio, matrix_lr, embedding_lr, window_pattern, description)
        best = get_best_result()
        print(f"\nIteration {iteration}/{max_iterations}. Best: val_bpb={best.val_bpb:.4f if best else 999}")

if __name__ == "__main__":
    main()
