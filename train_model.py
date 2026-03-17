#!/usr/bin/env python3
"""
Batch training with Graph Attention Network for velocity prediction.

Train/val/test: 70/15/15 temporal split.
Features: fisheye-compressed close history per edge + 4 metadata = 24 per edge.
Target: next-bar velocity per edge (log return, continuous).
Architecture: Graph Attention Network with message passing.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from accel_model import fisheye_boundaries, fisheye_sample, GraphAttentionNetwork
from coin_graph import CoinGraph

# ── hyperparams ──
Y_DEPTH = 200     # bars of close history per edge
X_PIXELS = 20     # fisheye output buckets per edge
CURVATURE = 2.0
EDGE_DIM = 128
NODE_DIM = 128
N_MESSAGE_PASSES = 2
EPOCHS = 50
BATCH = 32
LR = 3e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def build_dataset(graph: CoinGraph):
    """Pre-compute fisheye features and velocity targets for all bars."""
    n_bars = len(graph.common_timestamps)
    edges = list(graph.edge_names)
    n_edges = len(edges)

    # Build node indices
    node_set = set()
    for base, quote in edges:
        node_set.add(base)
        node_set.add(quote)
    node_names = sorted(list(node_set))
    node_to_idx = {n: i for i, n in enumerate(node_names)}
    n_nodes = len(node_names)

    # Build edge list with node indices
    edge_list = []
    for base, quote in edges:
        base_idx = node_to_idx[base]
        quote_idx = node_to_idx[quote]
        edge_list.append((base_idx, quote_idx))

    # 1. Collect all closes and velocities
    closes = np.zeros((n_bars, n_edges), dtype=np.float64)
    velocities = np.zeros((n_bars, n_edges), dtype=np.float32)

    for bar in range(n_bars):
        ts = graph.common_timestamps[bar]
        for ei, (base, quote) in enumerate(edges):
            df = graph.edges.get((base, quote))
            if df is None or ts not in df.index:
                continue
            row = df.loc[ts]
            c = float(row["close"])
            o = float(row["open"])
            closes[bar, ei] = c
            if o > 0:
                vel = np.log(c / o)
            else:
                vel = 0.0
            # Invert for reverse edges
            es = graph.edge_state.get((base, quote))
            if es and es.base != base:
                vel = -vel
            velocities[bar, ei] = vel

    # 2. Build fisheye features for bars [Y_DEPTH .. n_bars-2]
    #    Target = velocity at bar t+1
    boundaries = fisheye_boundaries(Y_DEPTH, X_PIXELS, CURVATURE)
    fee = graph.fee_rate

    xs, ys = [], []
    for t in range(Y_DEPTH, n_bars - 1):
        row_feat = []
        for ei in range(n_edges):
            history = closes[max(0, t - Y_DEPTH):t, ei]
            fv = fisheye_sample(history, boundaries)
            # Ensure we have exactly X_PIXELS values
            fv = fv[:X_PIXELS]
            while len(fv) < X_PIXELS:
                fv.append(0.0)
            row_feat.append(fv + [
                CURVATURE,
                Y_DEPTH / 1000.0,
                X_PIXELS / 100.0,
                fee
            ])
        xs.append(row_feat)
        ys.append(velocities[t + 1])  # next bar velocity

    X = np.array(xs, dtype=np.float32)  # (n_samples, n_edges, 24)
    Y = np.array(ys, dtype=np.float32)   # (n_samples, n_edges)
    return X, Y, edge_list, n_nodes, edges


def split(X, Y, train_frac=0.70, val_frac=0.15):
    n = len(X)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return (X[:t1], Y[:t1]), (X[t1:t2], Y[t1:t2]), (X[t2:], Y[t2:])


def train_epoch(model, loader, optimizer, criterion, edge_list, n_nodes):
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        # xb: (batch, n_edges, 24)
        # yb: (batch, n_edges)
        pred = model(xb, edge_list, n_nodes)  # (batch, n_edges)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, edge_list, n_nodes):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb, edge_list, n_nodes)
        loss = criterion(pred, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(yb.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    # Directional accuracy: did we predict the sign correctly?
    sign_match = ((preds > 0) == (targets > 0)).float().mean().item()
    return total_loss / n, sign_match


@torch.no_grad()
def simulate_test(model, X_test, Y_test, edge_list, n_nodes, fee=0.001):
    """Walk-forward on test set with frozen weights. Trade the best predicted edge each bar."""
    model.eval()
    capital = 100.0
    n_trades = 0
    wins = 0

    for i in range(len(X_test)):
        x = torch.tensor(X_test[i]).unsqueeze(0).to(DEVICE)
        pred = model(x, edge_list, n_nodes).squeeze().cpu().numpy()
        actual = Y_test[i]

        # Pick the edge with highest predicted velocity
        best_edge = int(np.argmax(pred))
        best_pred = pred[best_edge]

        if best_pred > fee:  # only trade if predicted return covers fee
            actual_vel = actual[best_edge]
            rate = actual_vel - fee
            capital *= (1 + rate)
            n_trades += 1
            if rate > 0:
                wins += 1
            if capital < 5.0:
                print(f"  Ruin at test bar {i}")
                break

    return capital, n_trades, wins


def main():
    print("Loading graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load()

    # Deduplicate edges (forward direction only)
    seen = set()
    unique_edges = []
    for b, q in graph.edges.keys():
        key = tuple(sorted([b, q]))
        if key not in seen:
            seen.add(key)
            unique_edges.append((b, q))
    graph.edge_names = unique_edges
    n_edges = len(unique_edges)
    print(f"Edges: {n_edges}, Bars: {n_bars}, Device: {DEVICE}")

    print("Building fisheye dataset (y_depth={}, x_pixels={})...".format(Y_DEPTH, X_PIXELS))
    X, Y, edge_list, n_nodes, edges = build_dataset(graph)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} edges × {X.shape[2]} features → {Y.shape[1]} targets")
    print(f"Graph: {n_nodes} nodes, {len(edge_list)} edges")

    # Stats
    up_frac = (Y > 0).mean()
    mean_vel = Y.mean()
    std_vel = Y.std()
    print(f"Target stats: mean={mean_vel:.6f}, std={std_vel:.6f}, up_frac={up_frac:.3f}")

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split(X, Y)
    print(f"Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
                          batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)),
                        batch_size=BATCH)

    model = GraphAttentionNetwork(
        edge_feature_dim=X.shape[2],  # 24 = X_PIXELS + 4
        edge_dim=EDGE_DIM,
        node_dim=NODE_DIM,
        n_message_passes=N_MESSAGE_PASSES
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GAT Model: {n_params:,} params")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_dir = 0.0
    best_state = None

    print(f"\nTraining {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, edge_list, n_nodes)
        val_loss, val_dir = evaluate(model, val_dl, criterion, edge_list, n_nodes)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        marker = ""
        if val_dir > best_val_dir:
            best_val_dir = val_dir
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"

        if (epoch + 1) % 5 == 0 or marker:
            print(f"  epoch {epoch+1:2d}: train_mse={train_loss:.6f} val_mse={val_loss:.6f} val_dir_acc={val_dir:.4f} lr={lr:.6f}{marker}")

    print(f"\nBest val directional accuracy: {best_val_dir:.4f}")
    print(f"Random baseline:               0.5000")
    print(f"Edge over random:              {best_val_dir - 0.5:.4f}")

    # Reload best
    if best_state:
        model.load_state_dict(best_state)

    # Per-edge directional accuracy on val
    model.eval()
    with torch.no_grad():
        vx = torch.tensor(X_val).to(DEVICE)
        vy = torch.tensor(Y_val)
        vp = model(vx, edge_list, n_nodes).cpu()
        print("\nPer-edge val directional accuracy:")
        for ei, edge in enumerate(edges):
            match = ((vp[:, ei] > 0) == (vy[:, ei] > 0)).float().mean().item()
            up = (vy[:, ei] > 0).float().mean().item()
            print(f"  {edge[0]:>5s}/{edge[1]:<5s}: dir_acc={match:.4f}  up_frac={up:.3f}")

    # Test simulation
    print(f"\nSimulating on {len(X_test)} test bars...")
    capital, n_trades, wins = simulate_test(model, X_test, Y_test, edge_list, n_nodes, fee=graph.fee_rate)
    print(f"  Start: $100.00  End: ${capital:.2f}  Gain: {capital - 100:+.2f}%")
    print(f"  Trades: {n_trades}  Wins: {wins}  Win rate: {wins/max(n_trades,1)*100:.1f}%")
    if n_trades > 0:
        print(f"  Avg gain/trade: {(capital/100 - 1)*100/n_trades:+.4f}%")

    torch.save(best_state, "velocity_model.pt")
    print("\nSaved velocity_model.pt")


if __name__ == "__main__":
    main()
