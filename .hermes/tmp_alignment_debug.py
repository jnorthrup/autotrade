import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coin_graph import CoinGraph
from hrm_model import HierarchicalReasoningModel


def summarize_bar_alignment(n_bars=8):
    g = CoinGraph(fee_rate=0.001)
    n = g.load(lookback_days=30, min_partners=3, max_partners=5, exchange='coinbase', skip_fetch=True)
    m = HierarchicalReasoningModel(n_edges=len(g.edges), h_dim=4, z_dim=4, y_depth=20, x_pixels=8, curvature=2.0, prediction_depth=1)
    edges = list(g.edges.keys())
    m.register_edges(edges)
    edge = edges[0]
    rows = []
    for bar_idx in range(min(n_bars, n)):
        edge_accels, edge_velocities, hit_ptt, hit_stop = g.update(bar_idx)
        buf_before_predict = len(m._close_buffer.get(edge, []))
        pred_queue_before = len(m._prediction_queue.get(edge, []))
        pred_feature_sum = None
        if bar_idx >= m.prediction_depth:
            m.predict(g, bar_idx)
            q = m._prediction_queue[edge]
            pred_feature_sum = round(sum(q[-1]['fisheye']), 8) if q else None
        buf_before_update = len(m._close_buffer.get(edge, []))
        if bar_idx >= m.prediction_depth * 2:
            loss = m.update(g, edge_accels, bar_idx, hit_ptt=hit_ptt, hit_stop=hit_stop)
        else:
            loss = None
        buf_after_update = len(m._close_buffer.get(edge, []))
        rows.append({
            'bar': bar_idx,
            'buf_before_predict': buf_before_predict,
            'pred_queue_before': pred_queue_before,
            'pred_feature_sum': pred_feature_sum,
            'buf_before_update': buf_before_update,
            'buf_after_update': buf_after_update,
            'hit_ptt': bool(hit_ptt.get(edge, False)),
            'hit_stop': bool(hit_stop.get(edge, False)),
            'loss': None if loss is None else round(loss, 6),
        })
    return rows

rows = summarize_bar_alignment()
for r in rows:
    print(r)

print('\nLABEL STATS')
g = CoinGraph(fee_rate=0.001)
n = g.load(lookback_days=30, min_partners=3, max_partners=5, exchange='coinbase', skip_fetch=True)
edge_counts = {'ptt': 0, 'stop': 0, 'neutral': 0, 'total': 0}
for bar_idx in range(min(500, n)):
    edge_accels, edge_velocities, hit_ptt, hit_stop = g.update(bar_idx)
    for e in g.edges:
        if e not in edge_velocities:
            continue
        p = bool(hit_ptt.get(e, False))
        s = bool(hit_stop.get(e, False))
        edge_counts['total'] += 1
        if p:
            edge_counts['ptt'] += 1
        elif s:
            edge_counts['stop'] += 1
        else:
            edge_counts['neutral'] += 1
print(edge_counts)
print({k: round(v / edge_counts['total'], 4) for k, v in edge_counts.items() if k != 'total'})
