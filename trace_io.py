"""Trace all value transfers through model pipeline.
Runs the harness with full IO logging to find blockers."""
import sys, os
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from cache import CoinGraph
from wallet import SimWallet
from showdown import _run_agent_harness, _canonical_pair_edges
from model import HierarchicalReasoningModel

LOG = []
def log(tag, **kw):
    entry = {"tag": tag, **kw}
    LOG.append(entry)
    parts = [f"[{tag}]"]
    for k, v in kw.items():
        if isinstance(v, dict):
            parts.append(f"{k}={v}")
        elif isinstance(v, (list, np.ndarray)):
            if len(str(v)) > 120:
                parts.append(f"{k}=len={len(v)}")
            else:
                parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    print("  ".join(parts), flush=True)


def make_trending_graph(n_bars=500, slope=0.5):
    graph = CoinGraph(fee_rate=0.001)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="5min")
    prices = [100.0 + i * slope for i in range(n_bars)]
    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000.0] * n_bars,
    }, index=timestamps)
    graph.add_product_frame("binance", "BTC-USDT", df)
    graph.common_timestamps = sorted(df.index)
    graph.all_pairs = ["binance:BTC-USDT"]
    return graph


def trace_harness():
    graph = make_trending_graph(500, slope=0.5)
    capital = 100.0
    y_depth = 50

    model = HierarchicalReasoningModel(
        n_edges=0,
        learning_rate=1e-3,
        y_depth=y_depth,
        x_pixels=64,
        curvature=0.3,
        h_dim=32,
        z_dim=32,
        prediction_depth=5,
        H_layers=2, L_layers=2, H_cycles=2, L_cycles=2,
        device="cpu",
        stochastic_fisheye=False,
    )
    model.register_edges(_canonical_pair_edges(graph))
    edge_names = model.edge_names

    wallet = SimWallet(["BTC", "USDT"], value_asset="USDT", initial_balances={"USDT": capital})

    # Wrap model methods with logging
    _orig_up = model.update_prices
    _orig_pred = model.predict
    _orig_upd = model.update
    _orig_force = model.force_update

    def _lup(graph, bar_idx):
        result = _orig_up(graph, bar_idx)
        for e in result:
            c = model._get_closes(graph, e, bar_idx)
            oc = model._obs_count(graph, e, bar_idx)
            log("update_prices", bar=bar_idx, edge=str(e), closes_len=len(c),
                close_last=f"{c[-1]:.4f}" if len(c) else "EMPTY", obs_count=oc,
                enough=len(c) >= model.y_depth)
        return result

    def _lpred(graph, bar_idx=-1):
        ready = [e for e in model._observed_edges_for_bar
                 if len(model._get_closes(graph, e, bar_idx)) >= model.y_depth]
        log("predict", bar=bar_idx, observed=[str(e) for e in model._observed_edges_for_bar],
            ready=[str(e) for e in ready], yd=model.y_depth)
        if not ready:
            return {}
        result = _orig_pred(graph, bar_idx)
        for e, v in result.items():
            q = model._prediction_queue.get(e, [])
            last_q = q[-1] if q else {}
            log("predict_result", bar=bar_idx, edge=str(e),
                frac=f"{v[0]:.4f}", bid=f"{v[2]:.4f}",
                queue_len=len(q), obs_count=last_q.get('observation_count'),
                start_price=last_q.get('start_price'))
        return result

    def _lupd(graph, actual_accels, bar_idx=-1, actual_velocities=None, hit_ptt=None, hit_stop=None):
        for e in actual_accels:
            frame = model._matured_prediction_frame(e, graph)
            oc = model._obs_count(graph, e, bar_idx)
            if frame:
                delta = oc - frame['observation_count']
                log("maturity", bar=bar_idx, edge=str(e), obs_now=oc,
                    obs_frame=frame['observation_count'], delta=delta,
                    matured=delta >= model.prediction_depth, pd=model.prediction_depth)
            else:
                log("maturity", bar=bar_idx, edge=str(e), obs_now=oc, frame="NONE")
        result = _orig_upd(graph, actual_accels, bar_idx, actual_velocities, hit_ptt, hit_stop)
        if bar_idx is not None and bar_idx % 50 == 0:
            log("update", bar=bar_idx, loss=result, n_accels=len(actual_accels))
        return result

    def _lforce(graph, edge_accels, bar_idx, **kw):
        log("force_update", bar=bar_idx, n_accels=len(edge_accels))
        result = _orig_force(graph, edge_accels, bar_idx, **kw)
        log("force_update_result", bar=bar_idx, loss=result)
        return result

    model.update_prices = _lup
    model.predict = _lpred
    model.update = _lupd
    model.force_update = _lforce

    log("init", y_depth=y_depth, edges=[str(e) for e in edge_names], capital=capital)
    print("=" * 80)
    print("TRACE: full IO logging — 500 bars trending")
    print("=" * 80)

    results = _run_agent_harness(
        graph,
        [(model, wallet)],
        sorted_bars=list(range(len(graph.common_timestamps))),
        capital=capital,
        use_profit_loss=False,
        print_every=5000,
    )

    pnl, dd = results[0]
    print(f"\n{'=' * 80}")
    print(f"FINAL: pnl={pnl:.2f} dd={dd:.2f}")

    # Summary
    tags = defaultdict(int)
    for e in LOG:
        tags[e["tag"]] += 1
    print(f"\n--- CALL COUNTS ---")
    for tag, count in sorted(tags.items()):
        print(f"  {tag}: {count}")

    # Find first predict with results and first update
    first_pred = first_upd = None
    for e in LOG:
        if e["tag"] == "predict_result" and first_pred is None:
            first_pred = e.get("bar")
        if e["tag"] == "update" and e.get("loss") is not None and first_upd is None:
            first_upd = e.get("bar")
    print(f"\nFirst predict with result: bar {first_pred}")
    print(f"First update with loss: bar {first_upd}")

    # Show maturity delta history
    print(f"\n--- MATURITY DELTA HISTORY (first 30) ---")
    for e in LOG:
        if e["tag"] == "maturity":
            print(f"  bar={e.get('bar')} edge={e.get('edge')} obs_now={e.get('obs_now')} "
                  f"obs_frame={e.get('obs_frame')} delta={e.get('delta')} "
                  f"matured={e.get('matured')} pd={e.get('pd')}")
            if len([x for x in LOG if x["tag"] == "maturity" and x is e]) > 30:
                break

    # Count maturity checks
    maturity_entries = [e for e in LOG if e["tag"] == "maturity"]
    matured_yes = sum(1 for e in maturity_entries if e.get("matured") == True)
    matured_no = sum(1 for e in maturity_entries if e.get("matured") == False)
    print(f"\nMatured YES: {matured_yes}  NO: {matured_no}")


if __name__ == "__main__":
    trace_harness()
