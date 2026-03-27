#!/usr/bin/env python3
"""End-to-end smoke test: load data -> train -> predict -> save/load checkpoint."""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_showdown import run_training
from coin_graph import CoinGraph
from hrm_model import HierarchicalReasoningModel


def main():
    print("=== Smoke Test: 50-bar training ===")

    # 1. Load graph
    print("Loading graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load(
        lookback_days=30,
        min_partners=3,
        max_partners=5,
        exchange="coinbase",
        skip_fetch=True,
    )
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges, {n_bars} bars")
    assert n_bars > 0, "No bars loaded"
    assert len(graph.edges) > 0, "No edges found"

    # 2. Create model
    model = HierarchicalReasoningModel(
        n_edges=len(graph.edges),
        h_dim=4,
        z_dim=4,
        y_depth=200,
        x_pixels=20,
        curvature=2.0,
        prediction_depth=1,
    )
    model.register_edges(list(graph.edges.keys()))
    print(f"Model params: {sum(p.numel() for p in model._model.parameters())}")

    # 3. Train on 50 bars
    end_bar = min(n_bars, 50)
    total_loss, n_updates, _, loss_history = run_training(
        graph, model,
        start_bar=0,
        end_bar=end_bar,
        print_every=25,
    )
    avg_loss = total_loss / n_updates if n_updates > 0 else float("inf")
    print(f"Training done: avg_loss={avg_loss:.6f}, n_updates={n_updates}")
    assert n_updates > 0, f"No training updates (n_updates={n_updates})"
    assert avg_loss < 3.0, f"Loss too high: {avg_loss:.6f}"
    assert len(loss_history) > 0, "No loss history recorded"

    # 4. Save and load checkpoint round-trip
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save(tmp_path)
        print(f"Saved checkpoint to {tmp_path}")

        model2 = HierarchicalReasoningModel(
            n_edges=len(graph.edges),
            h_dim=4,
            z_dim=4,
            y_depth=200,
            x_pixels=20,
            curvature=2.0,
            prediction_depth=1,
        )
        model2.register_edges(list(graph.edges.keys()))
        model2.load(tmp_path)
        print("Loaded checkpoint into fresh model")

        assert model2.h_dim == model.h_dim, f"h_dim mismatch: {model2.h_dim} != {model.h_dim}"
        assert model2.z_dim == model.z_dim, f"z_dim mismatch: {model2.z_dim} != {model.z_dim}"
        assert model2.x_pixels == model.x_pixels, f"x_pixels mismatch"
        assert model2.y_depth == model.y_depth, f"y_depth mismatch"
    finally:
        os.unlink(tmp_path)

    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
