"""KlineView: Stateless observable kline rendering pipeline.

Direct port of mp-superproject's KlineViewUtil.decorateView → pancake pipeline.

Each call to `decorate_view(edge, graph, bar_idx, wallet)` assembles a fresh feature
vector from the pandas DataFrame row at bar_idx. No buffers. No accumulation.
Pure function of (graph, bar_idx, edge, wallet).

Architecture (mirrors mp-superproject):
  Simulation.muxTrackedAssets  →  kline_view.decorate_view
  MuxIo.assembleRow            →  joins wallet state + time + pancaked OHLCV
  KlineViewUtil.pancake        →  flatten N rows × W cols → 1D branded vector
  KlineViewUtil.decorateView   →  full row: wallet + context + pancake + time
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# pancake: flatten N rows × W cols → 1D vector (branded column/rowIndex)
# Direct port of KlineViewUtil.pancake from mp-superproject
# ---------------------------------------------------------------------------

def pancake(rows: np.ndarray, col_names: List[str]) -> Dict[str, float]:
    """Flatten N rows × W cols OHLCV array into branded 1D dict.

    Like mp-superproject: ``srcRow.left[x.rem(width)] t2 "colName/rowIndex"``
    Returns dict like {"Open/0": 100.0, "High/0": 101.0, ..., "Close/3": 98.5}
    """
    if rows.size == 0:
        return {f"{c}/{i}": 0.0 for i in range(1) for c in col_names}
    n_rows, n_cols = rows.shape
    result = {}
    for row_idx in range(n_rows):
        for col_idx, col_name in enumerate(col_names):
            val = float(rows[row_idx, col_idx]) if col_idx < n_cols else 0.0
            result[f"{col_name}/{row_idx}"] = val if np.isfinite(val) else 0.0
    return result


# ---------------------------------------------------------------------------
# horizon: compressed index mapping like mp-superproject's horizon()
# ---------------------------------------------------------------------------

def horizon(y: int, horizon_width: int, total_rows: int) -> int:
    """Map compressed position y to actual row index in the window.

    Like mp-superproject: maps [0..horizon_width) → most recent rows,
    with early positions mapping to older data.
    """
    if total_rows <= horizon_width:
        return min(y, total_rows - 1)
    # Compress: newer data gets more resolution
    return max(0, total_rows - horizon_width + y)


# ---------------------------------------------------------------------------
# kline_ohlcv_slice: extract N rows of OHLCV from graph DataFrame
# ---------------------------------------------------------------------------

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
OHLCV_BRAND = ["Open", "High", "Low", "Close", "Volume"]


def kline_ohlcv_slice(
    graph: Any,
    edge: Tuple[str, ...],
    bar_idx: int,
    horizon_width: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    """Extract the last `horizon_width` rows of OHLCV for an edge ending at bar_idx.

    Returns (rows_array shape=[horizon_width, 5], branded_col_names).
    This is the equivalent of mp-superproject's ``shallow = Cursor(horizonDepth) { y -> curs at horizon(y, ...) }``
    followed by ``shallow["Open", "High", "Low", "Close", "Volume"]``.
    """
    if bar_idx < 0 or bar_idx >= len(getattr(graph, "common_timestamps", [])):
        return np.zeros((horizon_width, 5), dtype=np.float32), OHLCV_BRAND

    ts = graph.common_timestamps[bar_idx]
    df = getattr(graph, "edges", {}).get(edge)
    if df is None:
        return np.zeros((horizon_width, 5), dtype=np.float32), OHLCV_BRAND

    # Get rows up to and including ts
    loc = df.index.get_indexer([ts], method="ffill")
    end_idx = loc[0] if loc[0] >= 0 else len(df) - 1
    start_idx = max(0, end_idx - horizon_width + 1)
    window_df = df.iloc[start_idx:end_idx + 1]

    # Get inverted prices if needed
    inverted = getattr(graph, "edge_is_inverted", {}).get(edge, False)
    rows = []
    for _, row in window_df.iterrows():
        if hasattr(graph, "edge_price_components"):
            comps = graph.edge_price_components(edge, row)
            rows.append([
                comps.get("open", 0.0),
                comps.get("high", 0.0),
                comps.get("low", 0.0),
                comps.get("close", 0.0),
                float(row.get("volume", 0.0)) if not inverted else float(row.get("volume", 0.0)),
            ])
        else:
            rows.append([
                float(row.get("open", 0.0)),
                float(row.get("high", 0.0)),
                float(row.get("low", 0.0)),
                float(row.get("close", 0.0)),
                float(row.get("volume", 0.0)),
            ])

    if not rows:
        return np.zeros((horizon_width, 5), dtype=np.float32), OHLCV_BRAND

    arr = np.array(rows, dtype=np.float32)
    # Pad to horizon_width if shorter
    if len(arr) < horizon_width:
        pad = np.zeros((horizon_width - len(arr), 5), dtype=np.float32)
        # Fill padding with first row's close for continuity
        if len(arr) > 0:
            pad[:, 3] = arr[0, 3]  # close
            pad[:, 0] = arr[0, 3]  # open = close
            pad[:, 1] = arr[0, 3]  # high = close
            pad[:, 2] = arr[0, 3]  # low = close
        arr = np.vstack([pad, arr])

    return arr[-horizon_width:], OHLCV_BRAND


# ---------------------------------------------------------------------------
# is_live_bar: detect real vs forward-filled duplicate bars
# ---------------------------------------------------------------------------

def is_live_bar(graph: Any, edge: Tuple[str, ...], bar_idx: int) -> bool:
    """True if this bar has actual candle data (not forward-filled filler).

    A filler bar has: open == high == low == close AND volume == 0.
    This is what the reindex+ffill creates in cache.py lines 1223-1248.
    """
    if bar_idx < 0 or bar_idx >= len(getattr(graph, "common_timestamps", [])):
        return False
    ts = graph.common_timestamps[bar_idx]
    df = getattr(graph, "edges", {}).get(edge)
    if df is None or ts not in df.index:
        return False
    row = df.loc[ts]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    o, h, l, c, v = (
        float(row.get("open", 0.0)),
        float(row.get("high", 0.0)),
        float(row.get("low", 0.0)),
        float(row.get("close", 0.0)),
        float(row.get("volume", 0.0)),
    )
    # If inverted, get the actual stored values before inversion
    # (forward-fill happens on stored data, so check stored values)
    if abs(o - c) < 1e-10 and abs(h - c) < 1e-10 and abs(l - c) < 1e-10 and v < 1e-10:
        return False
    return True


# ---------------------------------------------------------------------------
# bar_features: inline velocity/range/volume features from OHLCV row
# ---------------------------------------------------------------------------

def bar_features(o: float, h: float, l: float, c: float, v: float) -> Dict[str, float]:
    """Compute per-bar features inline. No buffer needed.

    These are the same features graph.update() computes (velocity, range)
    but done statelessly from the OHLCV row itself.
    """
    features = {}
    features["velocity"] = math.log(c / o) if o > 0 else 0.0
    features["range"] = (h - l) / o if o > 0 else 0.0
    features["upper_wick"] = (h - max(o, c)) / o if o > 0 else 0.0
    features["lower_wick"] = (min(o, c) - l) / o if o > 0 else 0.0
    features["body"] = (c - o) / o if o > 0 else 0.0
    features["volume"] = v
    features["is_doji"] = 1.0 if abs(c - o) < (h - l) * 0.1 and (h - l) > 0 else 0.0
    return features


# ---------------------------------------------------------------------------
# decorate_view: the main stateless observable (mirrors KlineViewUtil.decorateView)
# ---------------------------------------------------------------------------

def decorate_view(
    graph: Any,
    edge: Tuple[str, ...],
    bar_idx: int,
    *,
    horizon_width: int = 20,
    wallet: Any = None,
) -> Dict[str, float]:
    """Assemble a complete feature dict for one edge at one bar.

    Direct port of mp-superproject's KlineViewUtil.decorateView:
    - pancaked OHLCV (N rows × 5 cols → 1D branded)
    - wallet free balance for this edge's assets
    - bar-level features (velocity, range, wicks, volume)
    - time cyclical features (DateShed-style)
    - live/filler flag

    STATELESS: no buffers, no accumulation. Pure function of inputs.
    """
    result = {}

    # 1. Pancaked OHLCV (KlineViewUtil.pancake equivalent)
    ohlcv_rows, col_names = kline_ohlcv_slice(graph, edge, bar_idx, horizon_width)
    result.update(pancake(ohlcv_rows, col_names))

    # 2. Per-bar features from the most recent candle
    if ohlcv_rows.shape[0] > 0 and ohlcv_rows.shape[1] >= 5:
        last = ohlcv_rows[-1]
        result.update({
            f"feat/{k}": v
            for k, v in bar_features(
                float(last[0]), float(last[1]),
                float(last[2]), float(last[3]),
                float(last[4])
            ).items()
        })

    # 3. Live bar indicator
    result["is_live"] = 1.0 if is_live_bar(graph, edge, bar_idx) else 0.0

    # 4. Wallet state (walletFree equivalent from MuxIo.assembleRow)
    if wallet is not None:
        _, base_asset, quote_asset = _parse_edge(edge)
        base_bal = wallet.balance(base_asset)
        quote_bal = wallet.balance(quote_asset)
        result["wallet/base_free"] = base_bal.free
        result["wallet/quote_free"] = quote_bal.free
        result["wallet/base_locked"] = base_bal.sim_locked
        result["wallet/quote_locked"] = quote_bal.sim_locked

    # 5. Time features (DateShed normalizeInstant equivalent)
    from model import time_cyclical_features
    time_feats = time_cyclical_features(bar_idx)
    time_names = [
        "time/sin_day", "time/cos_day",
        "time/sin_hour", "time/cos_hour",
        "time/sin_week", "time/cos_week",
        "time/sin_day2", "time/cos_day2",
        "time/sin_day3", "time/cos_day3",
        "time/bar_in_day", "time/bar_in_hour",
        "time/sin_day4", "time/cos_day4",
        "time/sin_day5", "time/cos_day5",
        "time/sin_hour2", "time/cos_hour2",
    ]
    for name, val in zip(time_names, time_feats):
        result[name] = val

    return result


def _parse_edge(edge):
    if len(edge) == 2:
        return None, str(edge[0]), str(edge[1])
    if len(edge) == 3:
        return str(edge[0]), str(edge[1]), str(edge[2])
    raise ValueError(f"Unsupported edge shape: {edge!r}")


# ---------------------------------------------------------------------------
# decorate_view_vector: same as decorate_view but returns ordered numpy array
# ---------------------------------------------------------------------------

# Brand cache: column order is fixed after first call
_BRAND_CACHE: Optional[List[str]] = None


def decorate_view_vector(
    graph: Any,
    edge: Tuple[str, ...],
    bar_idx: int,
    *,
    horizon_width: int = 20,
    wallet: Any = None,
) -> Tuple[np.ndarray, List[str]]:
    """Same as decorate_view but returns (numpy_vector, column_names).

    Column order is fixed across calls (branded like mp-superproject).
    """
    global _BRAND_CACHE

    feat_dict = decorate_view(graph, edge, bar_idx, horizon_width=horizon_width, wallet=wallet)

    if _BRAND_CACHE is None:
        _BRAND_CACHE = sorted(feat_dict.keys())

    # Ensure all cached keys exist
    vec = np.array([feat_dict.get(k, 0.0) for k in _BRAND_CACHE], dtype=np.float32)
    return vec, _BRAND_CACHE


def reset_brand_cache():
    """Reset the brand cache. Call when graph structure changes."""
    global _BRAND_CACHE
    _BRAND_CACHE = None


# ---------------------------------------------------------------------------
# batch_decorate: vectorized over all edges for one bar (like muxTrackedAssets)
# ---------------------------------------------------------------------------

def batch_decorate(
    graph: Any,
    edges: List[Tuple[str, ...]],
    bar_idx: int,
    *,
    horizon_width: int = 20,
    wallet: Any = None,
) -> Tuple[np.ndarray, List[str]]:
    """Decorate all edges for one bar. Returns (batch_array, col_names).

    Equivalent to mp-superproject's Simulation.muxTrackedAssets() followed by
    per-asset decorateView.
    """
    vectors = []
    col_names = None
    for edge in edges:
        vec, names = decorate_view_vector(graph, edge, bar_idx, horizon_width=horizon_width, wallet=wallet)
        vectors.append(vec)
        if col_names is None:
            col_names = names
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(vectors), col_names or []
