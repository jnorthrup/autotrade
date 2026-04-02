import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coin_graph import CoinGraph
from hrm_model import HierarchicalReasoningModel
from graph_showdown import run_training


def collect_targets(graph, limit_bars=300):
    counts = {'ptt': 0, 'stop': 0, 'neutral': 0, 'total': 0}
    for bar_idx in range(min(limit_bars, len(graph.common_timestamps))):
        edge_accels, edge_velocities, hit_ptt, hit_stop = graph.update(bar_idx)
        for edge in graph.edges:
            if edge not in edge_velocities:
                continue
            p = bool(hit_ptt.get(edge, False))
            s = bool(hit_stop.get(edge, False))
            counts['total'] += 1
            if p:
                counts['ptt'] += 1
            elif s:
                counts['stop'] += 1
            else:
                counts['neutral'] += 1
    return counts


def probe_predictions(graph, model, eval_start=200, eval_bars=50):
    preds = []
    labels = []
    end = min(len(graph.common_timestamps), eval_start + eval_bars)
    for bar_idx in range(end):
        edge_accels, edge_velocities, hit_ptt, hit_stop = graph.update(bar_idx)
        if bar_idx >= model.prediction_depth:
            out = model.predict(graph, bar_idx)
            if bar_idx >= eval_start:
                for edge, (f, p, s) in out.items():
                    preds.append((float(f), float(p), float(s)))
                    labels.append((1.0 if hit_ptt.get(edge, False) else 0.0,
                                   1.0 if hit_stop.get(edge, False) else 0.0,
                                   1.0 if hit_ptt.get(edge, False) else 0.0 if hit_stop.get(edge, False) else 0.5))
        if bar_idx >= model.prediction_depth * 2:
            model.update(graph, edge_accels, bar_idx, hit_ptt=hit_ptt, hit_stop=hit_stop)
    if not preds:
        return {}
    frac = [p[0] for p in preds]
    ptt = [p[1] for p in preds]
    stop = [p[2] for p in preds]
    frac_t = [l[2] for l in labels]
    ptt_t = [l[0] for l in labels]
    stop_t = [l[1] for l in labels]
    def mse(a, b):
        return sum((x-y)**2 for x,y in zip(a,b))/len(a)
    return {
        'n': len(preds),
        'frac_mean': statistics.mean(frac),
        'ptt_mean': statistics.mean(ptt),
        'stop_mean': statistics.mean(stop),
        'frac_target_mean': statistics.mean(frac_t),
        'ptt_target_mean': statistics.mean(ptt_t),
        'stop_target_mean': statistics.mean(stop_t),
        'frac_mse': mse(frac, frac_t),
        'ptt_mse': mse(ptt, ptt_t),
        'stop_mse': mse(stop, stop_t),
    }


graph = CoinGraph(fee_rate=0.001)
n_bars = graph.load(lookback_days=30, min_partners=3, max_partners=5, exchange='coinbase', skip_fetch=True)
print('target_counts', collect_targets(graph, 300))

graph = CoinGraph(fee_rate=0.001)
n_bars = graph.load(lookback_days=30, min_partners=3, max_partners=5, exchange='coinbase', skip_fetch=True)
model = HierarchicalReasoningModel(n_edges=len(graph.edges), h_dim=4, z_dim=4, y_depth=200, x_pixels=20, curvature=2.0, prediction_depth=1)
model.register_edges(list(graph.edges.keys()))
print('before_train', probe_predictions(graph, model, eval_start=50, eval_bars=25))

graph = CoinGraph(fee_rate=0.001)
n_bars = graph.load(lookback_days=30, min_partners=3, max_partners=5, exchange='coinbase', skip_fetch=True)
model = HierarchicalReasoningModel(n_edges=len(graph.edges), h_dim=4, z_dim=4, y_depth=200, x_pixels=20, curvature=2.0, prediction_depth=1)
model.register_edges(list(graph.edges.keys()))
loss, updates, _, history = run_training(graph, model, start_bar=0, end_bar=300, print_every=1000)
print('train_loss', loss/updates if updates else None, 'updates', updates, 'loss_hist_len', len(history))

graph = CoinGraph(fee_rate=0.001)
n_bars = graph.load(lookback_days=30, min_partners=3, max_partners=5, exchange='coinbase', skip_fetch=True)
print('after_train', probe_predictions(graph, model, eval_start=50, eval_bars=25))
