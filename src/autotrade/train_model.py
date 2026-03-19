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
    """
    Builds symmetric dataset mirroring past fisheye lens into future horizons.
    Resolution of memory == Resolution of foresight.
    """
    n_bars = len(graph.common_timestamps)
    edges = list(graph.edge_names)
    n_edges = len(edges)
    
    # 1. Collect all close prices and log returns
    closes = np.zeros((n_bars, n_edges))
    one_bar_rets = np.zeros((n_bars, n_edges))
    
    for bar in range(n_bars):
        ts = graph.common_timestamps[bar]
        for ei, edge in enumerate(edges):
            df = graph.edges.get(edge)
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            closes[bar, ei] = row["close"]
            # raw log return
            ret = np.log(row["close"] / row["open"]) if row["open"] > 0 else 0
            # handle reverse edge direction
            es = graph.edge_state.get(edge)
            if es and es.base != edge[0]:
                ret = -ret
            one_bar_rets[bar, ei] = ret

    # 2. Dynamic Symmetric Perspective (Fisheye Mirror)
    # The horizons are derived from the fisheye curvature
    # This ensures prediction resolution matches memory resolution.
    boundaries = fisheye_boundaries(Y_DEPTH, X_PIXELS, CURVATURE)
    xs, ys = [], []
    
    # Max future lookahead is the symmetric reflection of the history depth
    max_h = boundaries[-1]
    
    for t in range(Y_DEPTH, n_bars - max_h):
        row_feat = []
        staged_targets = []
        
        for ei in range(n_edges):
            # INPUT: Fisheye Memory
            # History window [t-Y_DEPTH : t]
            h = closes[t-Y_DEPTH:t, ei]
            fv = fisheye_sample(h, boundaries)
            # Metadata injection (curves + physical fee)
            row_feat.append(fv + [CURVATURE, Y_DEPTH/1000, X_PIXELS/100, graph.fee_rate])
            
            # OUTPUT: Symmetric Reflection (Foresight)
            # Prediction windows match input bucket widths
            edge_targets = []
            prev_b = 0
            for b in boundaries:
                # Target is cumulative return over the reflected window
                target_window_ret = np.sum(one_bar_rets[t + 1 + prev_b : t + 1 + b, ei])
                edge_targets.append(target_window_ret)
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
    n_bars = graph.load()
    
    # 1. Dataset Generation (Symmetric)
    # Deduplicate edges (forward direction only)
    seen = set()
    unique_edges = []
    for b, q in graph.edges.keys():
        key = tuple(sorted([b, q]))
        if key not in seen:
            seen.add(key)
            unique_edges.append((b, q))
    graph.edge_names = unique_edges
    
    print(f"Edges: {len(unique_edges)}, Bars: {n_bars}")
    X, Y = build_dataset(graph)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} edges × {X.shape[2]} features → {Y.shape[2]} symmetric targets")
    
    # 2. Train/Val Split
    split = int(len(X) * 0.8)
    x_train, y_train = X[:split], Y[:split]
    x_val, y_val = X[split:], Y[split:]
    
    # 3. Model & Optimizer
    model = HRM_ANE(hidden_size=EDGE_DIM, x_pixels=X_PIXELS)
    optimizer = optim.Adam(learning_rate=LR)
    loss_fn = HyperbolicLoss()
    
    best_val_acc = 0.0
    best_params = None
    
    # 4. Loop
    print(f"\nTraining {EPOCHS} epochs on ANE...")
    for epoch in range(EPOCHS):
        t0 = time.time()
        
        # Train
        train_loss = train_epoch(model, x_train, y_train, optimizer, loss_fn)
        
        # Eval
        val_loss, val_acc = evaluate(model, x_val, y_val, loss_fn)
        
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = tree_map(lambda p: p.copy(), model.parameters())
            marker = " *"
            
        print(f"Epoch {epoch+1:2d} | loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f} | {time.time()-t0:.1f}s{marker}")

    print(f"\nBest val directional accuracy: {best_val_acc:.4f}")
    
    # 5. Save Model
    if best_params:
        model.update(best_params)
        model.save_weights("velocity_model.mlx")
        print("Saved best weights to velocity_model.mlx")

if __name__ == "__main__":
    main()
