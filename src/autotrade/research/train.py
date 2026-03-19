"""
5-Minute Physical Trainer (Autoresearch Deliverable).
Optimizes HRM-ANE architecture against the 5-minute coin graph.
"""

import time
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, List, Tuple, Optional

from ..ane_model import HRM_ANE, fisheye_boundaries, fisheye_sample
from ..coin_graph import CoinGraph

# --- Physics Constants (5-Minute Intervals) ---
BAR_SECONDS = 300 

# --- Dynamic Hyperparameters (Pruned by auto.py) ---
CURVATURE = 2.802414141683337
HYPERBOLIC_POWER = 2.8041072416472863
H_CYCLES = 4
L_CYCLES = 3
Y_DEPTH = 200
X_PIXELS = 20
LR = 3e-4
EPOCHS = 20
BATCH = 32

class HyperbolicLoss:
    def __init__(self, power=2.0):
        self.power = power

    def __call__(self, pred, target):
        weight_intensity = mx.abs(target) ** self.power + 0.01
        mse = mx.square(pred - target) * weight_intensity
        agreement = mx.var(pred, axis=-1, keepdims=True)
        correct_mask = (mx.sign(pred) == mx.sign(target)).astype(mx.float32)
        precision_reward = mx.mean(agreement * mx.mean(correct_mask, axis=-1, keepdims=True) * mx.mean(weight_intensity, axis=-1, keepdims=True))
        return mx.mean(mse) + 0.5 * precision_reward

def build_dataset(graph: CoinGraph):
    """Symmetric mirroring of 5-minute intervals."""
    n_bars = len(graph.common_timestamps)
    edges = list(graph.edge_names)
    n_edges = len(edges)
    
    closes = np.zeros((n_bars, n_edges))
    one_bar_rets = np.zeros((n_bars, n_edges))
    
    for bar in range(n_bars):
        ts = graph.common_timestamps[bar]
        for ei, edge in enumerate(edges):
            df = graph.edges.get(edge)
            if df is None or ts not in df.index: continue
            row = df.loc[ts]
            closes[bar, ei] = row["close"]
            ret = np.log(row["close"] / row["open"]) if row["open"] > 0 else 0
            es = graph.edge_state.get(edge)
            if es and es.base != edge[0]: ret *= -1
            one_bar_rets[bar, ei] = ret

    boundaries = fisheye_boundaries(Y_DEPTH, X_PIXELS, CURVATURE)
    xs, ys = [], []
    max_h = boundaries[-1]
    
    for t in range(Y_DEPTH, n_bars - max_h, 5): # Stride to reduce correlation
        row_feat, staged_targets = [], []
        for ei in range(n_edges):
            h = closes[t-Y_DEPTH:t, ei]
            fv = fisheye_sample(h, boundaries)
            row_feat.append(fv + [CURVATURE, Y_DEPTH/1000, X_PIXELS/100, graph.fee_rate])
            
            edge_targets = []
            prev_b = 0
            for b in boundaries:
                edge_targets.append(np.sum(one_bar_rets[t+1+prev_b : t+1+b, ei]))
                prev_b = b
            staged_targets.append(edge_targets)
        xs.append(row_feat)
        ys.append(staged_targets)
    return mx.array(xs), mx.array(ys)

def main():
    print(f"Physical 5m Research Seed | Curvature={CURVATURE} Power={HYPERBOLIC_POWER}")
    graph = CoinGraph()
    graph.load(lookback_days=30) # 1 month of 5m bars for fast iteration
    
    # Prune to top 20 edges for research speed
    graph.edge_names = list(graph.edges.keys())[:20]
    
    X, Y = build_dataset(graph)
    split = int(len(X) * 0.8)
    x_train, y_train = X[:split], Y[:split]
    x_val, y_val = X[split:], Y[split:]
    
    model = HRM_ANE(hidden_size=128, x_pixels=X_PIXELS, H_cycles=H_CYCLES, L_cycles=L_CYCLES)
    optimizer = optim.Adam(learning_rate=LR)
    loss_fn = HyperbolicLoss(power=HYPERBOLIC_POWER)
    
    def loss_grad_fn(model, x, y):
        return loss_fn(model(x), y)
    
    val_grad_fn = nn.value_and_grad(model, loss_grad_fn)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        for i in range(0, len(indices), BATCH):
            idx = mx.array(indices[i:i+BATCH])
            loss, grads = val_grad_fn(model, x_train[idx], y_train[idx])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
        
        # Eval accuracy on 1-bar horizon
        pred = model(x_val)
        acc = mx.mean((mx.sign(pred[:,:,0]) == mx.sign(y_val[:,:,0])).astype(mx.float32)).item()
        if acc > best_val_acc: best_val_acc = acc
        print(f"Epoch {epoch+1} | Val Acc: {acc:.4f}")

    # Output metric for auto.py
    print(f"Final val_acc: {best_val_acc:.6f}")

if __name__ == "__main__":
    main()
