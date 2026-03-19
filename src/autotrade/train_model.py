import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from typing import Dict, List, Optional, Tuple, Any

from .ane_model import HRM_ANE, fisheye_boundaries, fisheye_sample
from .coin_graph import CoinGraph

# ── hyperparams ──
Y_DEPTH = 200
X_PIXELS = 20
CURVATURE = 2.0
EDGE_DIM = 128
EPOCHS = 50
BATCH = 32
LR = 3e-4

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
            one_bar_rets[bar, ei] = np.log(row["close"] / row["open"]) if row["open"] > 0 else 0
            es = graph.edge_state.get(edge)
            if es and es.base != edge[0]: one_bar_rets[bar, ei] *= -1

    boundaries = fisheye_boundaries(Y_DEPTH, X_PIXELS, CURVATURE)
    xs, ys = [], []
    max_h = boundaries[-1]
    
    for t in range(Y_DEPTH, n_bars - max_h):
        row_feat = []
        staged_targets = []
        for ei in range(n_edges):
            # Input features
            h = closes[t-Y_DEPTH:t, ei]
            fv = fisheye_sample(h, boundaries)
            row_feat.append(fv + [CURVATURE, Y_DEPTH/1000, X_PIXELS/100, graph.fee_rate])
            
            # Symmetric future targets
            edge_targets = []
            prev_b = 0
            for b in boundaries:
                edge_targets.append(np.sum(one_bar_rets[t+1+prev_b : t+1+b, ei]))
                prev_b = b
            staged_targets.append(edge_targets)
            
        xs.append(row_feat)
        ys.append(staged_targets)
        
    return mx.array(xs), mx.array(ys)

def train_epoch(model, x_train, y_train, optimizer, loss_fn):
    def batch_loss(model, x, y):
        return loss_fn(model(x), y)
    
    loss_and_grad_fn = nn.value_and_grad(model, batch_loss)
    
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    
    total_loss = 0
    for i in range(0, len(indices), BATCH):
        batch_idx = mx.array(indices[i:i+BATCH])
        x, y = x_train[batch_idx], y_train[batch_idx]
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
    return total_loss / (len(indices) // BATCH)

def evaluate(model, x_val, y_train, loss_fn):
    pred = model(x_val)
    loss = loss_fn(pred, y_train)
    # accuracy on the 1-bar horizon
    acc = mx.mean((mx.sign(pred[:,:,0]) == mx.sign(y_train[:,:,0])).astype(mx.float32))
    return loss.item(), acc.item()

def main():
    print("Apple Silicon Autotrade (MLX + ANE)")
    graph = CoinGraph()
    graph.load()
    
    # Simple edge selection for seed
    graph.edge_names = list(graph.edges.keys())[:20]
    
    X, Y = build_dataset(graph)
    split = int(len(X) * 0.8)
    x_train, y_train = X[:split], Y[:split]
    x_val, y_val = X[split:], Y[split:]
    
    model = HRM_ANE(hidden_size=EDGE_DIM, x_pixels=X_PIXELS)
    optimizer = optim.Adam(learning_rate=LR)
    loss_fn = HyperbolicLoss()
    
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss = train_epoch(model, x_train, y_train, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, x_val, y_val, loss_fn)
        print(f"Epoch {epoch+1} | Loss: {train_loss:.6f} | Val Acc: {val_acc:.4f} | {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
