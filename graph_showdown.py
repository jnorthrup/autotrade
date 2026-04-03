#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import zstandard as zstd

from hrm_model import HierarchicalReasoningModel
from coin_graph import (
    CoinGraph,
    EdgeState,
    NodeState,
    DEFAULT_MIN_PAIR_COVERAGE,
)
from config import Config
from candle_cache import (
    CandleCache,
    WS_SNAPSHOT_TARGET_CANDLES,
    _floor_time_to_granularity,
    _utc_isoformat,
    _utc_now_naive,
)

import duckdb
from pool_client import PoolClient, ensure_pool_running, pool_is_running

# --- Pool routing helpers ---
_pool_client_instance = None


def _use_pool() -> bool:
    """Check if the pool server is available."""
    return pool_is_running()


def _use_pool_for_db(db_path: str) -> bool:
    """Only route through the shared pool for the canonical on-disk DB."""
    try:
        if not db_path or db_path == ":memory:":
            return False
        return _use_pool() and Path(db_path).resolve() == Path(Config.DB_PATH).resolve()
    except Exception:
        return False


def _pool() -> PoolClient:
    """Get or create the singleton PoolClient."""
    global _pool_client_instance
    if _pool_client_instance is None:
        _pool_client_instance = PoolClient()
    return _pool_client_instance


def _resolve_experiments_db_path(training_db_path: Optional[str]) -> str:
    """Keep experiment metadata out of the live candle cache by default."""
    return str(Path(training_db_path or DEFAULT_TRAINING_DB_PATH))

# Square cube progression: hidden_size, always powers of 4
SQUARE_CUBE_SIZES = [4, 16, 64, 256]
PLATEAU_WINDOW = 100
PLATEAU_THRESHOLD = 1e-5
PLATEAU_PATIENCE = 3

# Growth cycle for square cube: which dimension leads next
# Cycle: h (hidden_size leads) -> H (H_layers catches up) -> L (L_layers catches up) -> h (cubed, cycle repeats)
GROWTH_CYCLE = ['h', 'H', 'L']
DEFAULT_TRAINING_DB_PATH = "training.duckdb"
WALK_FORWARD_VALIDATION_FRACTION = 0.2
DRAWTHROUGH_MAX_PASSES = 6


def _next_square_cube_size(size: int) -> Optional[int]:
    """Return the next allowed 4^k size, or None at the ceiling."""
    if size not in SQUARE_CUBE_SIZES:
        raise ValueError(f"Invalid square-cube size {size}; expected one of {SQUARE_CUBE_SIZES}")
    idx = SQUARE_CUBE_SIZES.index(size)
    if idx + 1 >= len(SQUARE_CUBE_SIZES):
        return None
    return SQUARE_CUBE_SIZES[idx + 1]


def _validate_square_cube_state(hidden_size: int, H_layers: int, L_layers: int):
    """Enforce powers-of-4 sizes with at most two distinct values."""
    sizes = (hidden_size, H_layers, L_layers)
    invalid = [size for size in sizes if size not in SQUARE_CUBE_SIZES]
    if invalid:
        raise ValueError(
            f"Square-cube state must use only {SQUARE_CUBE_SIZES}, got {sizes}"
        )
    if len(set(sizes)) > 2:
        raise ValueError(
            f"Square-cube state may use at most 2 distinct powers of 4, got {sizes}"
        )


def _apply_growth_step(
    model: HierarchicalReasoningModel,
    growth_dim: str,
    hidden_size: int,
    H_layers: int,
    L_layers: int,
) -> Tuple[int, int, int, bool]:
    """Apply exactly one 4× growth step if the scheduled dimension can grow."""
    _validate_square_cube_state(hidden_size, H_layers, L_layers)

    if growth_dim == 'h':
        if _next_square_cube_size(hidden_size) is None:
            return hidden_size, H_layers, L_layers, False
        model.grow('h')
        hidden_size = model.h_dim
    elif growth_dim == 'H':
        if _next_square_cube_size(H_layers) is None:
            return hidden_size, H_layers, L_layers, False
        model.grow('H')
        H_layers = model.H_layers
    elif growth_dim == 'L':
        if _next_square_cube_size(L_layers) is None:
            return hidden_size, H_layers, L_layers, False
        model.grow('L')
        L_layers = model.L_layers
    else:
        raise ValueError(f"Unknown growth dimension: {growth_dim}")

    _validate_square_cube_state(hidden_size, H_layers, L_layers)
    return hidden_size, H_layers, L_layers, True


def _is_converged(losses: List[float]) -> bool:
    """Sustained plateau detection across PATIENCE windows."""
    n = len(losses)
    if n < PLATEAU_WINDOW * PLATEAU_PATIENCE:
        return False
    for i in range(PLATEAU_PATIENCE):
        chunk = losses[-PLATEAU_WINDOW * (PLATEAU_PATIENCE - i):]
        if len(chunk) < PLATEAU_WINDOW:
            continue
        recent = np.mean(chunk[-PLATEAU_WINDOW:])
        older = np.mean(chunk[:PLATEAU_WINDOW])
        if abs(recent - older) > PLATEAU_THRESHOLD:
            return False
    return True


def _training_status(total_loss: float, n_updates: int) -> str:
    if n_updates <= 0:
        return "no scored updates; warmup or maturity never completed"
    return f"avg_loss={total_loss / n_updates:.6f}, n_updates={n_updates}"


def _print_profile_summary(profile_stats: Dict[str, float]):
    print("Profile summary:")
    print(
        f"  device={profile_stats.get('device_type', 'unknown')} "
        f"bars={int(profile_stats.get('bars_processed', 0))} "
        f"updates={int(profile_stats.get('n_updates', 0))}"
    )
    print(
        f"  graph.update={profile_stats.get('graph_update_seconds', 0.0):.4f}s "
        f"update_prices={profile_stats.get('update_prices_seconds', 0.0):.4f}s "
        f"predict_prepare={profile_stats.get('predict_prepare_seconds', 0.0):.4f}s "
        f"predict_forward={profile_stats.get('predict_forward_seconds', 0.0):.4f}s"
    )
    print(
        f"  update_prepare={profile_stats.get('update_prepare_seconds', 0.0):.4f}s "
        f"update_forward_backward={profile_stats.get('update_forward_backward_seconds', 0.0):.4f}s "
        f"predict_edges={int(profile_stats.get('predict_edges', 0))} "
        f"update_edges={int(profile_stats.get('update_edges', 0))}"
    )


def _checkpoint_variant(path: str, tag: str) -> str:
    checkpoint = Path(path)
    suffix = checkpoint.suffix or ".pt"
    return str(checkpoint.with_name(f"{checkpoint.stem}_{tag}{suffix}"))


def _ensure_experiments_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            timestamp TIMESTAMP DEFAULT now(),
            val_bpb DOUBLE,
            params VARCHAR,
            bag_spec VARCHAR,
            growth_phase VARCHAR,
            model_cas VARCHAR
        )
        """
    )
    conn.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS model_cas VARCHAR")


def _ensure_model_winners_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_winners (
            checkpoint_sha256 VARCHAR PRIMARY KEY,
            created_at TIMESTAMP DEFAULT now(),
            checkpoint_path VARCHAR,
            checkpoint_type VARCHAR,
            val_loss DOUBLE,
            n_updates BIGINT,
            params_json VARCHAR,
            bag_spec_json VARCHAR,
            model_cas VARCHAR,
            checkpoint_size_bytes BIGINT,
            checkpoint_blob_zstd BLOB
        )
        """
    )
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS checkpoint_path VARCHAR")
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS checkpoint_type VARCHAR")
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS val_loss DOUBLE")
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS n_updates BIGINT")
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS params_json VARCHAR")
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS bag_spec_json VARCHAR")
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS model_cas VARCHAR")
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS checkpoint_size_bytes BIGINT")
    conn.execute("ALTER TABLE model_winners ADD COLUMN IF NOT EXISTS checkpoint_blob_zstd BLOB")


def _publish_model_winner(
    training_db_path: str,
    checkpoint_path: str,
    checkpoint_type: str,
    val_loss: float,
    n_updates: int,
    params: Dict,
    bag_spec: Dict,
    model_cas: Optional[str],
) -> str:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    raw_bytes = checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    blob_zstd = zstd.ZstdCompressor(level=9).compress(raw_bytes)

    with duckdb.connect(training_db_path) as conn:
        _ensure_model_winners_table(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO model_winners (
                checkpoint_sha256,
                created_at,
                checkpoint_path,
                checkpoint_type,
                val_loss,
                n_updates,
                params_json,
                bag_spec_json,
                model_cas,
                checkpoint_size_bytes,
                checkpoint_blob_zstd
            )
            VALUES (?, now(), ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                checkpoint_sha256,
                str(checkpoint.resolve()),
                checkpoint_type,
                val_loss,
                n_updates,
                json.dumps(params, sort_keys=True, default=str),
                json.dumps(bag_spec, sort_keys=True, default=str),
                model_cas,
                len(raw_bytes),
                blob_zstd,
            ],
        )

    print(
        f"[Winner publish] loss={val_loss:.6f} updates={n_updates} "
        f"sha256={checkpoint_sha256} db={training_db_path}"
    )
    return checkpoint_sha256


def _load_bag_subscriptions(bag_path: str) -> List[Dict[str, str]]:
    with open(bag_path, "r") as f:
        raw_entries = json.load(f)
    if not isinstance(raw_entries, list):
        raise ValueError(f"Bag file must contain a JSON list of subscriptions: {bag_path}")

    deduped: Dict[Tuple[str, str], Dict[str, str]] = {}
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Bag file entry {idx} must be an object with exchange and product_id: {entry!r}"
            )
        exchange = str(entry.get("exchange") or "").strip()
        product_id = str(entry.get("product_id") or "").strip()
        if not exchange or not product_id or "-" not in product_id:
            raise ValueError(
                f"Bag file entry {idx} must include explicit exchange and product_id: {entry!r}"
            )
        sub = {"exchange": exchange, "product_id": product_id}
        deduped[(sub["exchange"], sub["product_id"])] = sub
    return list(deduped.values())


def _group_subscriptions_by_exchange(subscriptions: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for sub in subscriptions:
        grouped.setdefault(sub["exchange"], []).append(sub)
    return grouped


def _subscription_key(sub: Dict[str, str]) -> Tuple[str, str]:
    return str(sub["exchange"]), str(sub["product_id"])


def _format_product_group(product_ids: List[str]) -> str:
    by_quote: Dict[str, List[str]] = {}
    passthrough: List[str] = []

    for product_id in sorted(set(product_ids)):
        parts = product_id.split("-", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            passthrough.append(product_id)
            continue
        base, quote = parts
        by_quote.setdefault(quote, []).append(base)

    formatted: List[str] = []
    ordered_quotes = sorted(by_quote, key=lambda quote: (-len(set(by_quote[quote])), quote))
    for quote in ordered_quotes:
        bases = sorted(set(by_quote[quote]))
        if len(bases) == 1:
            formatted.append(f"{bases[0]}-{quote}")
        else:
            formatted.append("{" + ",".join(bases) + "}" + f"-{quote}")

    formatted.extend(passthrough)
    return ",".join(formatted)


def _format_subscription_keys(subscriptions: List[Dict[str, str]]) -> str:
    grouped: Dict[str, List[str]] = {}
    for sub in subscriptions:
        exchange, product_id = _subscription_key(sub)
        grouped.setdefault(exchange, []).append(product_id)
    parts = []
    for exchange in sorted(grouped):
        product_ids = _format_product_group(grouped[exchange])
        parts.append(f"{exchange}:{product_ids}")
    return ", ".join(parts)


def _checkpoint_source_summary(checkpoint_path: str) -> str:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return f"source=fresh path={checkpoint.resolve()} exists=False"
    try:
        state = torch.load(str(checkpoint), weights_only=False, map_location="cpu")
    except Exception as exc:
        return f"source=unreadable path={checkpoint.resolve()} exists=True error={exc}"
    if not isinstance(state, dict):
        return f"source=unknown path={checkpoint.resolve()} exists=True payload={type(state).__name__}"
    cas = state.get("model_cas", "n/a")
    checkpoint_type = state.get("checkpoint_type", "n/a")
    timestamp = state.get("checkpoint_timestamp", "n/a")
    h_dim = state.get("h_dim", "n/a")
    z_dim = state.get("z_dim", "n/a")
    H_layers = state.get("H_layers", "n/a")
    L_layers = state.get("L_layers", "n/a")
    return (
        f"source=checkpoint path={checkpoint.resolve()} exists=True "
        f"type={checkpoint_type} timestamp={timestamp} cas={cas} "
        f"h={h_dim} z={z_dim} H={H_layers} L={L_layers}"
    )


def _rotary_state_summary(model: HierarchicalReasoningModel) -> str:
    state = (model.h_dim, model.z_dim, model.H_layers, model.L_layers)
    distinct = sorted(set(state))
    return (
        f"h={model.h_dim} z={model.z_dim} H={model.H_layers} L={model.L_layers} "
        f"H_cycles={model.H_cycles} L_cycles={model.L_cycles} "
        f"powers={distinct} num_powers={len(distinct)}"
    )


def _print_coinbase_run_context(
    *,
    stage: str,
    bag_path: str,
    subscriptions: List[Dict[str, str]],
    checkpoint_path: str,
    training_db_path: Optional[str],
    learning_rate: float,
    y_depth: int,
    x_pixels: int,
    curvature: float,
    prediction_depth: int,
    granularity: str,
    lookback_days: int,
    device: str,
    live: bool,
    live_train_interval: int,
    live_http_repair_seconds: int,
    live_http_overlap_candles: int,
    live_idle_restart_seconds: int,
    model: Optional[HierarchicalReasoningModel] = None,
):
    print(
        f"[Coinbase run/{stage}] "
        f"bag_path={Path(bag_path).resolve()} pairs={len(subscriptions)} "
        f"subscriptions={_format_subscription_keys(subscriptions)}"
    )
    print(
        f"[Coinbase run/{stage}] "
        f"lr={learning_rate} y_depth={y_depth} x_pixels={x_pixels} curvature={curvature} "
        f"prediction_depth={prediction_depth} granularity={granularity} "
        f"lookback_days={lookback_days} device={device} live={live}"
    )
    print(
        f"[Coinbase run/{stage}] "
        f"live_train_interval={live_train_interval} "
        f"live_http_repair_seconds={live_http_repair_seconds} "
        f"live_http_overlap_candles={live_http_overlap_candles} "
        f"live_idle_restart_seconds={live_idle_restart_seconds} "
        f"training_db_path={Path(training_db_path).resolve() if training_db_path else 'None'}"
    )
    print(f"[Coinbase run/{stage}] {_checkpoint_source_summary(checkpoint_path)}")
    if model is not None:
        print(
            f"[Coinbase run/{stage}] "
            f"model_cas={model.model_cas_signature()} rotary={_rotary_state_summary(model)}"
        )


def _log_bag_coverage_summary(label: str, statuses: List[Dict[str, object]]) -> Tuple[int, int]:
    total = len(statuses)
    covered = [status for status in statuses if bool(status["covered"])]
    invalid = [status for status in statuses if not bool(status["covered"])]
    print(f"[{label}] contiguous={len(covered)}/{total} subscriptions")
    if invalid:
        preview = ", ".join(
            f"{status['exchange']}:{status['product_id']} ({float(status['coverage_ratio']):.3f})"
            for status in invalid[:8]
        )
        print(f"[{label}] incomplete sample: {preview}")
    return len(covered), len(invalid)


def _bag_surface_state(
    cache: CandleCache,
    subscriptions: List[Dict[str, str]],
    start: datetime,
    end: datetime,
    granularity: str = "300",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, str]], List[datetime]]:
    drawthrough = cache.query_bag_drawthrough(
        subscriptions,
        start,
        end,
        granularity=granularity,
    )
    surface_df = drawthrough["surface"]
    status_df = drawthrough["span_status"]
    common_support_df = drawthrough["common_support"]
    requested_keys = {_subscription_key(sub) for sub in subscriptions}
    if status_df.empty:
        missing = [
            {"exchange": exchange, "product_id": product_id}
            for exchange, product_id in sorted(requested_keys)
        ]
        return surface_df, status_df, common_support_df, missing, []

    present_keys = {
        (str(exchange), str(product_id))
        for exchange, product_id, actual_count in status_df[["exchange", "product_id", "actual_count"]].itertuples(index=False, name=None)
        if int(actual_count) > 0
    }
    missing = [
        {"exchange": exchange, "product_id": product_id}
        for exchange, product_id in sorted(requested_keys - present_keys)
    ]

    common_timestamps = [
        pd.Timestamp(ts).to_pydatetime()
        for ts, is_common in common_support_df[["timestamp", "is_common"]].itertuples(index=False, name=None)
        if bool(is_common)
    ] if not common_support_df.empty else []
    common_timestamps.sort()

    return surface_df, status_df, common_support_df, missing, common_timestamps


def _drawthrough_repair_window(
    cache: CandleCache,
    subscriptions: List[Dict[str, str]],
    *,
    exchange: str,
    start: datetime,
    end: datetime,
    granularity: str = "300",
    label: str = "drawthrough",
    max_passes: int = DRAWTHROUGH_MAX_PASSES,
) -> None:
    cache.repair_bag_drawthrough(
        subscriptions,
        start,
        end,
        granularity=granularity,
        max_passes=max_passes,
        log_prefix="[Coinbase week/live]",
        label=label,
    )


def _surface_union_timestamps(surface_df: pd.DataFrame) -> List[pd.Timestamp]:
    if surface_df.empty or "timestamp" not in surface_df.columns:
        return []
    return sorted(
        {
            pd.Timestamp(ts)
            for ts in surface_df["timestamp"].tolist()
            if not pd.isna(ts)
        }
    )


def _finalize_selected_graph_timestamps(
    graph: CoinGraph,
    *,
    surface_df: pd.DataFrame,
    common_timestamps: List[datetime],
) -> None:
    graph._align_timestamps()
    if graph.common_timestamps:
        return

    support_timestamps = sorted(pd.Timestamp(ts) for ts in common_timestamps)
    if support_timestamps:
        graph.common_timestamps = support_timestamps
        print(
            f"[SelectedGraph] no exact dataframe intersection; "
            f"using bag-surface common-support timeline ({len(support_timestamps)} bars)"
        )
        return

    union_timestamps = _surface_union_timestamps(surface_df)
    if union_timestamps:
        graph.common_timestamps = union_timestamps
        print(
            f"[SelectedGraph] no all-pair common bars; "
            f"using union timeline ({len(union_timestamps)} bars)"
        )
        return

    raise RuntimeError("Fixed Coinbase bag surface has no timestamps to replay")


def _load_selected_pair_graph(
    selected_subscriptions: List[Dict[str, str]],
    start: datetime,
    end: datetime,
    granularity: str = "300",
    fee_rate: float = 0.001,
    min_pair_coverage: float = DEFAULT_MIN_PAIR_COVERAGE,
) -> CoinGraph:
    cache = CandleCache(str(Config.DB_PATH))
    graph = CoinGraph(fee_rate=fee_rate, min_pair_coverage=min_pair_coverage)
    graph.cache = cache
    graph.bag_surface_name = None
    graph.bag_thresholds_view_name = None
    surface_df, _status_df, common_support_df, missing_subscriptions, common_timestamps = _bag_surface_state(
        cache,
        selected_subscriptions,
        start,
        end,
        granularity=granularity,
    )
    if missing_subscriptions:
        raise RuntimeError(
            "Fixed Coinbase bag surface is missing requested subscriptions: "
            f"{_format_subscription_keys(missing_subscriptions)}"
        )
    if surface_df.empty:
        raise RuntimeError("No candle data available for the requested bag surface")

    expected = max(1, int((end - start).total_seconds() // int(granularity)))
    counts_by_key = {
        (str(exchange), str(product_id)): int(count)
        for (exchange, product_id), count in surface_df.groupby(["exchange", "product_id"]).size().items()
    }
    materialized_subscriptions: List[Dict[str, str]] = []

    full_index = pd.date_range(start=start, end=end, freq=f"{granularity}s", inclusive="left")

    for (exchange, pid), df in surface_df.groupby(["exchange", "product_id"], sort=False):
        parts = pid.split("-", 1)
        if len(parts) != 2:
            continue
        df = (
            df.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .set_index("timestamp")
        )
        if df.empty:
            continue
        coverage = min(1.0, counts_by_key.get((exchange, pid), 0) / expected)

        missing_idx = full_index.difference(df.index)
        df = df.reindex(full_index)
        df["close"] = df["close"].ffill().bfill()
        df.loc[missing_idx, "open"] = df.loc[missing_idx, "close"]
        df.loc[missing_idx, "high"] = df.loc[missing_idx, "close"]
        df.loc[missing_idx, "low"] = df.loc[missing_idx, "close"]
        df.loc[missing_idx, "volume"] = 0.0

        graph.add_product_frame(exchange, pid, df, coverage=coverage)
        materialized_subscriptions.append({"exchange": exchange, "product_id": pid})

    graph.all_pairs = [f"{sub['exchange']}:{sub['product_id']}" for sub in materialized_subscriptions]
    graph.bag_subscriptions = materialized_subscriptions

    if "USD" in graph.nodes:
        graph.nodes.discard("USD")
        graph.nodes = {"USD"} | graph.nodes

    _finalize_selected_graph_timestamps(
        graph,
        surface_df=surface_df,
        common_timestamps=common_timestamps,
    )
    return graph


def _replay_margin_bars(model: HierarchicalReasoningModel) -> int:
    min_train_bars = max(model.y_depth + model.prediction_depth + 64, 256)
    min_validation_bars = max(model.prediction_depth + 32, 64)
    return max(model.y_depth + model.prediction_depth + 32, min_train_bars + min_validation_bars)


def plan_walk_forward_split(
    total_bars: int,
    model: HierarchicalReasoningModel,
    validation_fraction: float = WALK_FORWARD_VALIDATION_FRACTION,
) -> Optional[Tuple[int, int, int]]:
    min_train_bars = max(model.y_depth + model.prediction_depth + 64, 256)
    min_validation_bars = max(model.prediction_depth + 32, 64)
    if total_bars < (min_train_bars + min_validation_bars):
        return None

    validation_bars = max(min_validation_bars, int(total_bars * validation_fraction))
    validation_bars = min(validation_bars, total_bars - min_train_bars)
    if validation_bars < min_validation_bars:
        return None

    train_end_bar = total_bars - validation_bars
    if train_end_bar < min_train_bars:
        return None
    return 0, train_end_bar, total_bars


def _clone_graph_for_replay(graph: CoinGraph) -> CoinGraph:
    clone = CoinGraph(fee_rate=graph.fee_rate, min_pair_coverage=graph.min_pair_coverage)
    clone.all_pairs = list(graph.all_pairs)
    clone.bag_subscriptions = list(getattr(graph, "bag_subscriptions", []))
    clone.nodes = set(graph.nodes)
    clone.common_timestamps = list(graph.common_timestamps)
    for edge, df in graph.edges.items():
        clone.edges[edge] = df
        clone.edge_state[edge] = EdgeState()
    clone.edge_product_id = dict(getattr(graph, "edge_product_id", {}))
    clone.edge_is_inverted = dict(getattr(graph, "edge_is_inverted", {}))
    for node in graph.node_state:
        clone.node_state[node] = NodeState()
    clone.pair_coverage = dict(getattr(graph, "pair_coverage", {}))
    clone.pair_exchange = dict(getattr(graph, "pair_exchange", {}))
    return clone


def _clone_model_for_evaluation(
    model: HierarchicalReasoningModel,
    edge_names: List[Tuple[str, str]],
) -> HierarchicalReasoningModel:
    clone = HierarchicalReasoningModel(
        n_edges=len(edge_names),
        learning_rate=model._lr,
        y_depth=model.y_depth,
        x_pixels=model.x_pixels,
        curvature=model.curvature,
        h_dim=model.h_dim,
        z_dim=model.z_dim,
        prediction_depth=model.prediction_depth,
        H_layers=model.H_layers,
        L_layers=model.L_layers,
        H_cycles=model.H_cycles,
        L_cycles=model.L_cycles,
        device=model.device_preference,
    )
    clone.register_edges(list(edge_names))
    if model._model is not None and clone._model is not None:
        clone._model.load_state_dict(model._model.state_dict())
    return clone


def run_walk_forward_validation(
    graph: CoinGraph,
    trained_model: HierarchicalReasoningModel,
    validation_start_bar: int,
    end_bar: Optional[int] = None,
    warmup_start_bar: int = 0,
    print_every: int = 0,
) -> Tuple[Optional[float], int]:
    if end_bar is None:
        end_bar = len(graph.common_timestamps)
    if validation_start_bar >= end_bar:
        return None, 0

    eval_graph = _clone_graph_for_replay(graph)
    eval_model = _clone_model_for_evaluation(trained_model, list(graph.edges.keys()))

    total_loss = 0.0
    n_updates = 0
    active_predictions = {}
    total_pnl = 0.0
    max_pnl = 0.0
    max_drawdown = 0.0

    for bar_idx in range(warmup_start_bar, end_bar):
        edge_accels, edge_velocities, hit_ptt, hit_stop = eval_graph.update(bar_idx)
        if not edge_accels:
            continue

        eval_model.update_prices(eval_graph, bar_idx)
        if eval_model.ready_for_prediction(bar_idx):
            preds = eval_model.predict(eval_graph, bar_idx)
            active_predictions[bar_idx] = preds
            if bar_idx == end_bar - 1 and preds:
                ts_str = str(eval_graph.common_timestamps[bar_idx])
                signals = []
                for edge, (frac, ptt, stop) in preds.items():
                    if ptt > 0.55 or stop > 0.55:
                        signals.append(f"{edge[0]}-{edge[1]} B={ptt:.2f} S={stop:.2f}")
                if signals:
                    print(f"  [PaperTrade @ {ts_str}] " + " | ".join(signals[:6]) + (f" (+{len(signals)-6} more)" if len(signals) > 6 else ""))


        if not eval_model.ready_for_update(bar_idx, edge_accels):
            continue

        loss = eval_model.score(
            eval_graph,
            edge_accels,
            bar_idx,
            hit_ptt=hit_ptt,
            hit_stop=hit_stop,
        )

        mature_idx = bar_idx - eval_model.prediction_depth
        if mature_idx in active_predictions:
            matured_preds = active_predictions.pop(mature_idx)
            bar_pnl = 0.0
            for edge, (frac, ptt, stop) in matured_preds.items():
                if edge not in edge_velocities:
                    continue
                vel = edge_velocities[edge]
                if ptt > 0.55:
                    bar_pnl += vel - eval_graph.fee_rate
                elif stop > 0.55:
                    bar_pnl += -vel - eval_graph.fee_rate
            total_pnl += bar_pnl
            if total_pnl > max_pnl:
                max_pnl = total_pnl
            dd = max_pnl - total_pnl
            if dd > max_drawdown:
                max_drawdown = dd

        if loss is None or bar_idx < validation_start_bar:
            continue

        total_loss += loss
        n_updates += 1
        if print_every and n_updates % print_every == 0:
            avg_loss = total_loss / n_updates
            print(
                f"[WalkForward] bar={bar_idx} avg_loss={avg_loss:.6f} "
                f"updates={n_updates} pnl={total_pnl:.5f} max_dd={max_drawdown:.5f}"
            )

    if n_updates <= 0:
        return None, 0
    return total_loss / n_updates, n_updates


def run_training(graph: CoinGraph, model: HierarchicalReasoningModel, start_bar: int = 0,
                 end_bar: Optional[int] = None, print_every: int = 100,
                 loss_history: Optional[List[float]] = None,
                 profile_stats: Optional[Dict[str, float]] = None) -> Tuple[float, int, bool, List[float]]:
    """Train model on graph bars. Returns (total_loss, n_updates, early_stopped, loss_history)."""
    if end_bar is None:
        end_bar = len(graph.common_timestamps)
    
    if loss_history is None:
        loss_history = []

    model.set_profile_enabled(profile_stats is not None)
    
    ts_to_bar = {ts: i for i, ts in enumerate(graph.common_timestamps)}
    bars_with_data = set()
    for df in graph.edges.values():
        common_ts = set(df.index) & ts_to_bar.keys()
        bars_with_data.update(ts_to_bar[ts] for ts in common_ts)
    
    total_loss = 0.0
    n_updates = 0
    early_stopped = False
    graph_update_seconds = 0.0
    
    active_predictions = {}
    total_pnl = 0.0
    max_pnl = 0.0
    max_drawdown = 0.0

    sorted_bars = sorted(b for b in bars_with_data if start_bar <= b < end_bar)
    print(f"Training on {len(sorted_bars)} bars with data")
    
    loop_start = time.perf_counter() if profile_stats is not None else None
    for i, bar_idx in enumerate(sorted_bars):
        if bar_idx >= len(graph.common_timestamps):
            break

        graph_update_start = time.perf_counter() if profile_stats is not None else None
        edge_accels, edge_velocities, hit_ptt, hit_stop = graph.update(bar_idx)
        if graph_update_start is not None:
            graph_update_seconds += time.perf_counter() - graph_update_start
        
        if not edge_accels:
            continue

        model.update_prices(graph, bar_idx)
        
        if model.ready_for_prediction(bar_idx):
            preds = model.predict(graph, bar_idx)
            active_predictions[bar_idx] = preds
        
        if model.ready_for_update(bar_idx, edge_accels):
            loss = model.update(graph, edge_accels, bar_idx, hit_ptt=hit_ptt, hit_stop=hit_stop)
            if loss is not None:
                total_loss += loss
                n_updates += 1
                loss_history.append(loss)
        
        mature_idx = bar_idx - model.prediction_depth
        if mature_idx in active_predictions:
            matured_preds = active_predictions.pop(mature_idx)
            bar_pnl = 0.0
            for edge, (frac, ptt, stop) in matured_preds.items():
                if edge not in edge_velocities:
                    continue
                vel = edge_velocities[edge]
                if ptt > 0.55:
                    bar_pnl += vel - graph.fee_rate
                elif stop > 0.55:
                    bar_pnl += -vel - graph.fee_rate
            total_pnl += bar_pnl
            if total_pnl > max_pnl:
                max_pnl = total_pnl
            dd = max_pnl - total_pnl
            if dd > max_drawdown:
                max_drawdown = dd

        if (i % print_every == 0 and i > 0) or (i == len(sorted_bars) - 1):
            if n_updates > 0:
                avg_loss = total_loss / n_updates
                print(f"Train[{i+1}/{len(sorted_bars)}] Bar {bar_idx}: avg_loss={avg_loss:.6f}, updates={n_updates}, pnl={total_pnl:.5f}, max_dd={max_drawdown:.5f}")
            else:
                print(f"Train[{i+1}/{len(sorted_bars)}] Bar {bar_idx}: warmup, candle history queues filling ({i+1}/{model.y_depth})")

    if profile_stats is not None:
        profile_stats.clear()
        profile_stats.update(model.get_profile_stats())
        profile_stats['graph_update_seconds'] = graph_update_seconds
        profile_stats['bars_processed'] = float(len(sorted_bars))
        profile_stats['bars_with_data'] = float(len(sorted_bars))
        profile_stats['n_updates'] = float(n_updates)
        profile_stats['loop_seconds'] = time.perf_counter() - loop_start if loop_start is not None else 0.0
    
    return total_loss, n_updates, early_stopped, loss_history


def _list_all_binance_pairs(db_path: str) -> List[str]:
    """List all Binance-style pairs available in candle_cache (DuckDB)."""
    try:
        if _use_pool_for_db(db_path):
            rows = _pool().execute("SELECT DISTINCT product_id FROM candles WHERE exchange = 'binance'")
            pairs = [r[0] for r in rows]
        else:
            with duckdb.connect(db_path, read_only=True) as conn:
                rows = conn.execute("SELECT DISTINCT product_id FROM candles WHERE exchange = 'binance'").fetchall()
                pairs = [r[0] for r in rows]
        # Filter: must look like BASE-QUOTE with two parts
        pairs = [p for p in pairs if "-" in p and len(p.split("-", 1)) == 2]
        return sorted(set(pairs))
    except Exception as e:
        print(f"[_list_all_binance_pairs] error: {e}")
        return []


def _normalize_subscription_record(entry, default_exchange: Optional[str] = None) -> Optional[Dict[str, str]]:
    if isinstance(entry, dict):
        exchange = str(entry.get("exchange") or default_exchange or "").strip()
        product_id = entry.get("product_id") or entry.get("pair") or entry.get("symbol")
    elif isinstance(entry, str):
        raw = entry.strip()
        if not raw:
            return None
        if ":" in raw:
            maybe_exchange, maybe_product = raw.split(":", 1)
            if maybe_exchange and "-" in maybe_product:
                exchange = maybe_exchange.strip()
                product_id = maybe_product.strip()
            else:
                exchange = str(default_exchange or "").strip()
                product_id = raw
        else:
            exchange = str(default_exchange or "").strip()
            product_id = raw
    else:
        return None

    product_id = str(product_id or "").strip()
    if not exchange or "-" not in product_id:
        return None
    return {"exchange": exchange, "product_id": product_id}


def _list_all_exchange_subscriptions(db_path: str, exchange: Optional[str] = None) -> List[Dict[str, str]]:
    try:
        if _use_pool_for_db(db_path):
            if exchange is None:
                rows = _pool().execute("SELECT DISTINCT exchange, product_id FROM candles")
            else:
                rows = _pool().execute(
                    "SELECT DISTINCT exchange, product_id FROM candles WHERE exchange = ?",
                    [exchange],
                )
            rows = list(rows)
        else:
            with duckdb.connect(db_path, read_only=True) as conn:
                if exchange is None:
                    rows = conn.execute("SELECT DISTINCT exchange, product_id FROM candles").fetchall()
                else:
                    rows = conn.execute(
                        "SELECT DISTINCT exchange, product_id FROM candles WHERE exchange = ?",
                        [exchange],
                    ).fetchall()
        subscriptions = []
        for row in rows:
            if isinstance(row, tuple) and len(row) >= 2:
                sub = _normalize_subscription_record({"exchange": row[0], "product_id": row[1]})
                if sub is not None:
                    subscriptions.append(sub)
        deduped: Dict[Tuple[str, str], Dict[str, str]] = {}
        for sub in subscriptions:
            deduped[(sub["exchange"], sub["product_id"])] = sub
        return list(deduped.values())
    except Exception as e:
        print(f"[_list_all_exchange_subscriptions] error: {e}")
        return []


FIAT_CURRENCIES = {
    "USD", "USDT", "USDC", "EUR", "GBP", "SGD", "JPY", "BRL", "MXN",
    "TRY", "IDR", "PLN", "ARS", "ZAR", "UAH", "COP", "RUB", "NGN", "EURI",
}


def _compute_volatility_filter(
    db_path: str,
    all_pairs,
    lookback_days: int = 365,
    granularity: str = "300",
    min_velocity: float = 0.001,
) -> List:
    """Filter pairs by mean |velocity| (log close/open) and drop fiat-fiat edges.

    Returns pairs with mean |velocity| >= min_velocity and no fiat-fiat edges.
    """
    end = _utc_now_naive()
    start = end - timedelta(days=lookback_days)

    filtered = []
    normalized_inputs = []
    for entry in all_pairs or []:
        sub = _normalize_subscription_record(entry)
        if sub is None:
            continue
        normalized_inputs.append((entry, sub))

    use_pool_flag = _use_pool_for_db(db_path)
    pool = _pool() if use_pool_flag else None

    if use_pool_flag:
        for original, sub in normalized_inputs:
            pid = sub["product_id"]
            exchange = sub["exchange"]
            parts = pid.split("-", 1)
            if len(parts) != 2:
                continue
            base, quote = parts
            if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
                continue
            try:
                rows = pool.execute(
                    """SELECT AVG(ABS(LN(close / NULLIF(open, 0))))
                       FROM candles
                       WHERE exchange = ? AND product_id = ? AND granularity = ?
                         AND timestamp >= ? AND timestamp < ?
                         AND open > 0 AND close > 0""",
                    [exchange, pid, granularity, start, end],
                )
                mean_vel = rows[0][0] if rows and rows[0] and rows[0][0] is not None else 0.0
            except Exception:
                mean_vel = 0.0

            if mean_vel >= min_velocity:
                filtered.append(original)
    else:
        with duckdb.connect(db_path, read_only=True) as conn:
            for original, sub in normalized_inputs:
                pid = sub["product_id"]
                exchange = sub["exchange"]
                parts = pid.split("-", 1)
                if len(parts) != 2:
                    continue
                base, quote = parts
                if base in FIAT_CURRENCIES and quote in FIAT_CURRENCIES:
                    continue
                try:
                    row = conn.execute(
                        """SELECT AVG(ABS(LN(close / NULLIF(open, 0))))
                           FROM candles
                           WHERE exchange = ? AND product_id = ? AND granularity = ?
                             AND timestamp >= ? AND timestamp < ?
                             AND open > 0 AND close > 0""",
                        [exchange, pid, granularity, start, end],
                    ).fetchone()
                    mean_vel = row[0] if row and row[0] is not None else 0.0
                except Exception:
                    mean_vel = 0.0

                if mean_vel >= min_velocity:
                    filtered.append(original)

    if filtered and isinstance(filtered[0], dict):
        deduped: Dict[Tuple[str, str], Dict[str, str]] = {}
        for sub in filtered:
            deduped[(sub["exchange"], sub["product_id"])] = sub
        return list(deduped.values())
    return sorted(set(filtered))


# Model-size to bag-size scaling: 4->5, 16->20, 64->40, 256->80
_BAG_SIZE_SCALE = {4: 5, 16: 20, 64: 40, 256: 80}


def _stochastic_bag_sample(
    filtered_pairs: List[str],
    model_size: int,
    rng: random.Random,
    min_pairs: int = 5,
    max_pairs: Optional[int] = None,
) -> List[str]:
    """Sample a stochastic bag of pairs, size scaled by model dimensions.

    Bag size = _BAG_SIZE_SCALE.get(model_size, model_size * 3 / 4), clamped to
    [min_pairs, max_pairs or len(filtered_pairs)].
    """
    if not filtered_pairs:
        return []

    target = _BAG_SIZE_SCALE.get(model_size, max(min_pairs, model_size * 3 // 4))
    if max_pairs is not None:
        target = min(target, max_pairs)
    target = max(min_pairs, min(target, len(filtered_pairs)))

    # Build adjacency for connected subgraph sampling
    adj = _build_pair_adjacency(filtered_pairs)
    return _select_related_pairs(filtered_pairs, adj, target, rng)


def _build_pair_adjacency(all_pairs: List[str]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for pid in all_pairs:
        parts = pid.split("-", 1)
        if len(parts) != 2:
            continue
        for currency in parts:
            adj.setdefault(currency, []).append(pid)
    return adj


def _select_related_pairs(all_pairs: List[str], adj: Dict[str, List[str]],
                          n_pairs: int, rng: random.Random) -> List[str]:
    if n_pairs >= len(all_pairs):
        return list(all_pairs)

    currencies = list(adj.keys())
    if not currencies:
        return list(all_pairs)[:n_pairs]

    selected = set()
    seed_currency = rng.choice(currencies)
    frontier = [seed_currency]
    visited_currencies = {seed_currency}

    while len(selected) < n_pairs and frontier:
        curr = frontier.pop(0)
        candidates = [p for p in adj.get(curr, []) if p not in selected]
        rng.shuffle(candidates)
        for pid in candidates:
            if len(selected) >= n_pairs:
                break
            selected.add(pid)
            parts = pid.split("-", 1)
            for c in parts:
                if c not in visited_currencies:
                    visited_currencies.add(c)
                    frontier.append(c)

        if not frontier and len(selected) < n_pairs:
            remaining = [c for c in currencies if c not in visited_currencies]
            if remaining:
                new_seed = rng.choice(remaining)
                frontier.append(new_seed)
                visited_currencies.add(new_seed)

    return list(selected)


def _make_trial_graph(full_graph: CoinGraph, selected_pairs: List[str],
                      start_bar: int, end_bar: int) -> CoinGraph:
    from coin_graph import EdgeState, NodeState
    
    trial = CoinGraph(
        fee_rate=full_graph.fee_rate,
        min_pair_coverage=getattr(full_graph, "min_pair_coverage", DEFAULT_MIN_PAIR_COVERAGE),
    )
    source_exchange = None
    if getattr(full_graph, "bag_subscriptions", []):
        source_exchange = full_graph.bag_subscriptions[0]["exchange"]
    trial.all_pairs = list(selected_pairs)
    trial.bag_subscriptions = (
        [{"exchange": source_exchange, "product_id": pid} for pid in selected_pairs]
        if source_exchange is not None
        else []
    )

    pair_to_edges = {}
    for sub in getattr(full_graph, "bag_subscriptions", []):
        pid = sub["product_id"]
        exchange = sub["exchange"]
        parts = pid.split("-", 1)
        if len(parts) == 2:
            pair_to_edges[(exchange, pid)] = (exchange, parts[0], parts[1])

    for pid in selected_pairs:
        edge_key = pair_to_edges.get((source_exchange, pid)) if source_exchange is not None else None
        if edge_key is None:
            continue
        exchange, base, quote = edge_key
        for edge in [(exchange, base, quote), (exchange, quote, base)]:
            if edge in full_graph.edges:
                trial.edges[edge] = full_graph.edges[edge]
                trial.edge_state[edge] = EdgeState()
                if edge in getattr(full_graph, "edge_product_id", {}):
                    trial.edge_product_id[edge] = full_graph.edge_product_id[edge]
                if edge in getattr(full_graph, "edge_is_inverted", {}):
                    trial.edge_is_inverted[edge] = full_graph.edge_is_inverted[edge]
        trial.nodes.add(base)
        trial.nodes.add(quote)
        trial.node_state.setdefault(base, NodeState())
        trial.node_state.setdefault(quote, NodeState())

    if trial.edges and all(not callable(getattr(df, "index", None)) for df in trial.edges.values()):
        trial._align_timestamps()
        trial.common_timestamps = trial.common_timestamps[start_bar:end_bar]
    else:
        trial.common_timestamps = full_graph.common_timestamps[start_bar:end_bar]
    return trial


def run_autoresearch(graph: CoinGraph, db_path: str = 'candles.duckdb',
                     exchange: str = 'coinbase', pm_mode: str = 'single_asset',
                     device: str = 'auto',
                     checkpoint_path: Optional[str] = None,
                     training_db_path: Optional[str] = None):
    """
    Autoresearch with Square Cube Progression + Stochastic Bag Sampling.

    Starts with tiny HRM (hidden_size=4, H_layers=4, L_layers=4).
    Trains until plateau, then grows one dimension by 4× using rotational expansion.

    Each iteration:
      1. Load all available pairs from candle_cache (or fall back to graph.all_pairs)
      2. Volatility filter: drop low-velocity and fiat-fiat pairs
      3. Stochastic bag: randomly sample N pairs, size scaled by model dimensions
      4. Stochastic time window: randomly sample (start_bar, end_bar)

    Growth cycle: h -> H -> L -> h (hidden_size leads, then layers catch up)
    Square sizes: 4 -> 16 -> 64 -> 256 (always powers of 4)
    """
    print("Using HRMEdgePredictor for hierarchical reasoning")

    experiments_db_path = _resolve_experiments_db_path(training_db_path)
    if Path(experiments_db_path).resolve() == Path(db_path).resolve():
        print(
            "[Autoresearch] WARNING: experiments DB shares the candle cache path; "
            "this can block live candle writes."
        )
    conn = duckdb.connect(experiments_db_path)
    _ensure_experiments_table(conn)

    best_bpb = float('inf')
    best_params = None
    incumbent_checkpoint_path = checkpoint_path
    best_checkpoint_path = _checkpoint_variant(checkpoint_path, "best") if checkpoint_path else None
    grown_checkpoint_path = _checkpoint_variant(checkpoint_path, "grown") if checkpoint_path else None

    total_bars = len(graph.common_timestamps)
    rng = random.Random()

    # --- Stochastic bag: load exchange-qualified subscriptions, then volatility-filter ---
    all_db_subscriptions = _list_all_exchange_subscriptions(db_path, exchange=exchange)
    if not all_db_subscriptions:
        all_db_subscriptions = [
            sub for sub in getattr(graph, "bag_subscriptions", [])
            if sub.get("exchange") == exchange
        ]
    if all_db_subscriptions:
        print(f"[StochasticBag] Found {len(all_db_subscriptions)} subscriptions in candle_cache")
        filtered_subscriptions = _compute_volatility_filter(
            db_path,
            all_db_subscriptions,
            lookback_days=365 if exchange == "coinbase" else 1095,
            granularity="300",
            min_velocity=0.001,
        )
        print(f"[StochasticBag] After volatility filter: {len(filtered_subscriptions)} subscriptions")
        filtered_pairs = [sub["product_id"] for sub in filtered_subscriptions]
    else:
        # Fallback to the graph's own bag subscriptions or legacy discovery IDs.
        fallback_subscriptions = [
            sub for sub in getattr(graph, "bag_subscriptions", [])
            if sub.get("exchange") == exchange
        ]
        if not fallback_subscriptions:
            fallback_subscriptions = []
            for entry in graph.all_pairs:
                sub = _normalize_subscription_record(entry, default_exchange=exchange)
                if sub is not None and sub["exchange"] == exchange:
                    fallback_subscriptions.append(sub)
        filtered_pairs = [sub["product_id"] for sub in fallback_subscriptions]
        print(
            f"[StochasticBag] No DB subscriptions found, using graph bag/discovery "
            f"({len(filtered_pairs)} pairs)"
        )

    if not filtered_pairs:
        print("[StochasticBag] ERROR: no pairs available after filtering")
        return None

    total_pool = len(filtered_pairs)
    MIN_PAIRS = 5
    MAX_PAIRS = total_pool
    MIN_WINDOW_BARS = max(200, total_bars // 20)
    MAX_WINDOW_BARS = total_bars

    # Square cube state
    growth_idx = 0  # index into GROWTH_CYCLE
    hidden_size = SQUARE_CUBE_SIZES[0]  # start at 4
    H_layers = SQUARE_CUBE_SIZES[0]
    L_layers = SQUARE_CUBE_SIZES[0]
    current_h_dim = hidden_size
    phase = 0

    incumbent_checkpoint = Path(incumbent_checkpoint_path) if incumbent_checkpoint_path else None
    if incumbent_checkpoint is not None and incumbent_checkpoint.exists():
        seed_model = HierarchicalReasoningModel(device=device)
        seed_model.load(str(incumbent_checkpoint))
        hidden_size = seed_model.h_dim
        H_layers = seed_model.H_layers
        L_layers = seed_model.L_layers
        current_h_dim = hidden_size
        print(
            f"Resuming autoresearch from {incumbent_checkpoint_path}: "
            f"h={current_h_dim}, H={H_layers}, L={L_layers}"
        )

    _validate_square_cube_state(hidden_size, H_layers, L_layers)

    print(f"\nAutoresearch: {total_pool} filtered pairs, {total_bars} bars")
    print(f"Square Cube: hidden_size={hidden_size}, H_layers={H_layers}, L_layers={L_layers}")
    print(f"Bag scaling: 4->5, 16->20, 64->40, 256->80 pairs")
    print(f"Window: [{MIN_WINDOW_BARS}..{MAX_WINDOW_BARS}] bars")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            phase += 1
            progress = min(1.0, phase / 200.0)

            # --- Stochastic bag sampling scaled by current model size ---
            selected_pairs = _stochastic_bag_sample(
                filtered_pairs, current_h_dim, rng,
                min_pairs=MIN_PAIRS, max_pairs=MAX_PAIRS,
            )
            n_pairs = len(selected_pairs)

            window_ceil = int(MIN_WINDOW_BARS + (MAX_WINDOW_BARS - MIN_WINDOW_BARS) * progress)
            window_bars = rng.randint(MIN_WINDOW_BARS, max(MIN_WINDOW_BARS, window_ceil))
            window_bars = min(window_bars, total_bars)

            max_start = max(0, total_bars - window_bars)
            start_bar = rng.randint(0, max_start) if max_start > 0 else 0
            end_bar = start_bar + window_bars

            window_days = round(window_bars * 5 / (60 * 24), 1)

            trial_graph = _make_trial_graph(graph, selected_pairs, start_bar, end_bar)
            if not trial_graph.edges:
                print(f"Phase {phase}: empty graph, skipping")
                continue

            lr = 10 ** random.uniform(-4, -1.5)
            y_depth = random.choice([100, 200, 300, 400])
            x_pixels = random.choice([10, 15, 20, 30])
            curvature = random.uniform(0.5, 4.0)
            prediction_depth = random.choice([1, 2, 3, 5, 10])

            print()
            print(f"=== Phase {phase} (p={progress:.2f}) ===")
            print(f"  Square: hidden_size={current_h_dim}, H_layers={H_layers}, L_layers={L_layers}")
            print(f"  Bag: {n_pairs} pairs, {window_bars} bars ({window_days}d)")

            model = HierarchicalReasoningModel(
                n_edges=len(trial_graph.edges),
                learning_rate=lr,
                y_depth=y_depth,
                x_pixels=x_pixels,
                curvature=curvature,
                h_dim=current_h_dim,
                z_dim=current_h_dim,
                prediction_depth=prediction_depth,
                H_layers=H_layers,
                L_layers=L_layers,
                H_cycles=2,
                L_cycles=2,
                device=device,
            )
            model.register_edges(list(trial_graph.edges.keys()))
            if incumbent_checkpoint is not None and incumbent_checkpoint.exists():
                model.load(str(incumbent_checkpoint))
                hidden_size = model.h_dim
                H_layers = model.H_layers
                L_layers = model.L_layers
                current_h_dim = hidden_size
                _validate_square_cube_state(hidden_size, H_layers, L_layers)

            split = plan_walk_forward_split(len(trial_graph.common_timestamps), model)
            if split is None:
                print("  Skipping: not enough bars for walk-forward train/validation split")
                continue
            _, train_end_bar, validation_end_bar = split
            print(
                f"  Walk-forward: train=0:{train_end_bar}, "
                f"validate={train_end_bar}:{validation_end_bar}"
            )

            loss_history = []
            total_loss, n_updates, _, loss_history = run_training(
                trial_graph,
                model,
                start_bar=0,
                end_bar=train_end_bar,
                print_every=1000,
                loss_history=loss_history,
            )
            train_bpb = (total_loss / n_updates) if n_updates > 0 else None
            val_bpb, val_updates = run_walk_forward_validation(
                trial_graph,
                model,
                validation_start_bar=train_end_bar,
                end_bar=validation_end_bar,
                warmup_start_bar=0,
            )
            model_cas = model.model_cas_signature()
            if n_updates > 0 and incumbent_checkpoint_path:
                model.save(
                    incumbent_checkpoint_path,
                    checkpoint_type=f"autoresearch_phase_{phase}",
                )

            params = {
                'lr': lr, 'h_dim': current_h_dim,
                'y_depth': y_depth, 'x_pixels': x_pixels,
                'curvature': curvature, 'prediction_depth': prediction_depth,
                'H_layers': H_layers, 'L_layers': L_layers,
                'train_end_bar': train_end_bar,
                'validation_start_bar': train_end_bar,
                'validation_end_bar': validation_end_bar,
            }
            bag_spec = {
                'n_pairs': n_pairs, 'window_bars': window_bars,
                'window_days': window_days, 'start_bar': start_bar,
                'selection_policy': 'stochastic_historical_universe',
                'timestamp_policy': 'intersection',
                'exchange': exchange,
            }
            growth_dim = GROWTH_CYCLE[growth_idx]
            if val_bpb is None:
                print(
                    f"  train={_training_status(total_loss, n_updates)} "
                    f"walk_forward=n/a"
                )
            else:
                train_text = f"{train_bpb:.6f}" if train_bpb is not None else "n/a"
                print(
                    f"  train_bpb={train_text} "
                    f"walk_forward_bpb={val_bpb:.6f} "
                    f"(updates={val_updates}, best={best_bpb:.6f})"
                )

            if val_bpb is not None and val_bpb < best_bpb:
                best_bpb = val_bpb
                best_params = {**params, **bag_spec, 'growth_phase': f'{growth_dim}', 'model_cas': model_cas}
                print(f"  --> [NEW BEST] val_bpb: {best_bpb:.6f}")
                if best_checkpoint_path:
                    model.save(
                        best_checkpoint_path,
                        checkpoint_type=f"autoresearch_best_phase_{phase}",
                    )
                    if training_db_path:
                        _publish_model_winner(
                            training_db_path=training_db_path,
                            checkpoint_path=best_checkpoint_path,
                            checkpoint_type=f"autoresearch_best_phase_{phase}",
                            val_loss=val_bpb,
                            n_updates=val_updates,
                            params=params,
                            bag_spec=bag_spec,
                            model_cas=model_cas,
                        )

            _insert_sql = (
                "INSERT INTO experiments (timestamp, val_bpb, params, bag_spec, growth_phase, model_cas)"
                " VALUES (now(), ?, ?, ?, ?, ?)"
            )
            _insert_params = [val_bpb, str(params), str(bag_spec), growth_dim, model_cas]
            conn.execute(_insert_sql, _insert_params)

            # Check for convergence and trigger growth
            if _is_converged(loss_history):
                old_h = current_h_dim
                old_H = H_layers
                old_L = L_layers

                hidden_size, H_layers, L_layers, did_grow = _apply_growth_step(
                    model, growth_dim, hidden_size, H_layers, L_layers
                )
                current_h_dim = hidden_size

                if did_grow:
                    growth_idx = (growth_idx + 1) % len(GROWTH_CYCLE)
                    print(
                        f"\n  *** CONVERGED -> GROWTH: {growth_dim} "
                        f"[h={old_h}, H={old_H}, L={old_L}] -> "
                        f"[h={current_h_dim}, H={H_layers}, L={L_layers}]"
                    )
                    if incumbent_checkpoint_path:
                        model.save(
                            incumbent_checkpoint_path,
                            checkpoint_type=f"autoresearch_grown_phase_{phase}",
                        )
                    if grown_checkpoint_path:
                        model.save(
                            grown_checkpoint_path,
                            checkpoint_type=f"autoresearch_grown_phase_{phase}",
                        )
                else:
                    print(
                        f"\n  *** CONVERGED -> NO GROWTH: {growth_dim} already at max "
                        f"[h={current_h_dim}, H={H_layers}, L={L_layers}]"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        conn.close()

    if best_params:
        print(f"Best: val_bpb={best_bpb:.6f} with {best_params}")
    return best_params


def run_coinbase_week_http_then_live(
    bag_path: str,
    checkpoint_path: str,
    training_db_path: Optional[str],
    lookback_days: int,
    granularity: str,
    learning_rate: float,
    y_depth: int,
    x_pixels: int,
    curvature: float,
    prediction_depth: int,
    h_dim: int,
    z_dim: int,
    device: str,
    print_every: int,
    save_checkpoint: bool,
    live: bool,
    live_train_interval: int,
    live_http_repair_seconds: int,
    live_http_overlap_candles: int,
    live_idle_restart_seconds: int,
):
    selected_subscriptions = _load_bag_subscriptions(bag_path)
    if not selected_subscriptions:
        raise ValueError(f"No valid subscriptions found in {bag_path}")
    non_coinbase = [sub for sub in selected_subscriptions if sub["exchange"] != "coinbase"]
    if non_coinbase:
        raise ValueError(
            "coinbase-week-http-then-live requires an all-Coinbase bag; found: "
            f"{_format_subscription_keys(non_coinbase)}"
        )
    selected_by_exchange = _group_subscriptions_by_exchange(selected_subscriptions)
    _print_coinbase_run_context(
        stage="requested",
        bag_path=bag_path,
        subscriptions=selected_subscriptions,
        checkpoint_path=checkpoint_path,
        training_db_path=training_db_path,
        learning_rate=learning_rate,
        y_depth=y_depth,
        x_pixels=x_pixels,
        curvature=curvature,
        prediction_depth=prediction_depth,
        granularity=granularity,
        lookback_days=lookback_days,
        device=device,
        live=live,
        live_train_interval=live_train_interval,
        live_http_repair_seconds=live_http_repair_seconds,
        live_http_overlap_candles=live_http_overlap_candles,
        live_idle_restart_seconds=live_idle_restart_seconds,
    )

    cache = CandleCache(str(Config.DB_PATH))
    coinbase_validation = cache.validate_coinbase_products(
        [sub["product_id"] for sub in selected_subscriptions if sub["exchange"] == "coinbase"]
    )
    bad_coinbase = list(coinbase_validation["missing"]) + list(coinbase_validation["invalid"])
    if bad_coinbase:
        raise RuntimeError(
            "Coinbase bag contains products that are not live Coinbase Advanced spot products: "
            + ", ".join(bad_coinbase)
        )
    bootstrap = cache.bootstrap_database(granularity=granularity)
    print(
        f"[Coinbase week/live bootstrap] db={bootstrap['db_path']} "
        f"pool={bootstrap['pool_enabled']} "
        f"normalized={bootstrap['normalized_timestamps']} "
        f"purged={bootstrap['purged_future_rows']}"
    )
    ensure_pool_running(str(Config.DB_PATH))
    print(f"[Coinbase week/live bootstrap] duckdb_pool=ready db={Path(Config.DB_PATH).resolve()}")
    initial_end = _floor_time_to_granularity(_utc_now_naive(), granularity)
    initial_start = initial_end - timedelta(days=lookback_days)

    print(
        f"[Coinbase week/live] subscriptions={_format_subscription_keys(selected_subscriptions)} "
        f"pairs={len(selected_subscriptions)} "
        f"window={_utc_isoformat(initial_start)} -> {_utc_isoformat(initial_end)} "
        f"granularity={granularity} device={device}"
    )

    bootstrap_coverage = cache.bag_coverage_ratio(
        selected_subscriptions, initial_start, initial_end, granularity=granularity,
    )
    print(f"[Coinbase week/live] DB coverage={bootstrap_coverage:.1%} for {len(selected_subscriptions)} pairs")

    for exchange, subs in sorted(selected_by_exchange.items()):
        pairs = [sub["product_id"] for sub in subs]
        if exchange != "coinbase":
            raise NotImplementedError(f"Unsupported exchange in bag: {exchange}")

        if bootstrap_coverage < 0.95:
            print(
                f"[Coinbase week/live] ws snapshot seed "
                f"pairs={len(pairs)} target_bars={WS_SNAPSHOT_TARGET_CANDLES}"
            )
            cache.ws_snapshot(
                pairs,
                granularity=granularity,
                exchange=exchange,
                target_candles=WS_SNAPSHOT_TARGET_CANDLES,
            )

            _drawthrough_repair_window(
                cache,
                subs,
                exchange=exchange,
                start=initial_start,
                end=initial_end,
                granularity=granularity,
                label="drawthrough",
            )
        else:
            print(
                f"[Coinbase week/live] bootstrap skip (coverage={bootstrap_coverage:.1%}) "
                f"ws_snapshot and drawthrough not needed"
            )

    graph = _load_selected_pair_graph(
        selected_subscriptions,
        initial_start,
        initial_end,
        granularity=granularity,
    )
    if not graph.edges or not graph.common_timestamps:
        raise RuntimeError("No candle data available after HTTP backfill")
    materialized_keys = {_subscription_key(sub) for sub in graph.bag_subscriptions}
    requested_keys = {_subscription_key(sub) for sub in selected_subscriptions}
    if materialized_keys != requested_keys:
        missing = [
            {"exchange": exchange, "product_id": product_id}
            for exchange, product_id in sorted(requested_keys - materialized_keys)
        ]
        extra = [
            {"exchange": exchange, "product_id": product_id}
            for exchange, product_id in sorted(materialized_keys - requested_keys)
        ]
        details = []
        if missing:
            details.append(f"missing={_format_subscription_keys(missing)}")
        if extra:
            details.append(f"extra={_format_subscription_keys(extra)}")
        raise RuntimeError(
            "Fixed Coinbase bag must materialize exactly as requested; refusing to continue after bag drift. "
            + "; ".join(details)
        )
    selected_subscriptions = list(graph.bag_subscriptions)

    model = HierarchicalReasoningModel(
        n_edges=len(graph.edges),
        learning_rate=learning_rate,
        y_depth=y_depth,
        x_pixels=x_pixels,
        curvature=curvature,
        h_dim=h_dim,
        z_dim=z_dim,
        prediction_depth=prediction_depth,
        device=device,
    )
    model.register_edges(list(graph.edges.keys()))
    model.load(checkpoint_path)
    _print_coinbase_run_context(
        stage="loaded",
        bag_path=bag_path,
        subscriptions=selected_subscriptions,
        checkpoint_path=checkpoint_path,
        training_db_path=training_db_path,
        learning_rate=learning_rate,
        y_depth=model.y_depth,
        x_pixels=model.x_pixels,
        curvature=model.curvature,
        prediction_depth=model.prediction_depth,
        granularity=granularity,
        lookback_days=lookback_days,
        device=device,
        live=live,
        live_train_interval=live_train_interval,
        live_http_repair_seconds=live_http_repair_seconds,
        live_http_overlap_candles=live_http_overlap_candles,
        live_idle_restart_seconds=live_idle_restart_seconds,
        model=model,
    )

    print(
        f"[Coinbase week/live] training initial week: "
        f"bars={len(graph.common_timestamps)} edges={len(graph.edges)}"
    )
    initial_split = plan_walk_forward_split(len(graph.common_timestamps), model)
    if initial_split is None:
        raise RuntimeError("Not enough bars for walk-forward validation in fixed-bag live mode")
    _, initial_train_end_bar, initial_validation_end_bar = initial_split
    total_loss, n_updates, _, _ = run_training(
        graph,
        model,
        start_bar=0,
        end_bar=initial_train_end_bar,
        print_every=print_every,
    )
    avg_loss = (total_loss / n_updates) if n_updates > 0 else None
    walk_forward_loss, walk_forward_updates = run_walk_forward_validation(
        graph,
        model,
        validation_start_bar=initial_train_end_bar,
        end_bar=initial_validation_end_bar,
        warmup_start_bar=0,
    )
    best_loss = walk_forward_loss if walk_forward_loss is not None else float("inf")

    week_params = {
        "mode": "coinbase_week_http_then_live",
        "lr": learning_rate,
        "h_dim": model.h_dim,
        "z_dim": model.z_dim,
        "y_depth": model.y_depth,
        "x_pixels": model.x_pixels,
        "curvature": model.curvature,
        "prediction_depth": model.prediction_depth,
        "device": device,
        "granularity": granularity,
        "lookback_days": lookback_days,
        "train_end_bar": initial_train_end_bar,
        "validation_start_bar": initial_train_end_bar,
        "validation_end_bar": initial_validation_end_bar,
    }
    week_bag_spec = {
        "pairs": len(selected_subscriptions),
        "subscriptions": selected_subscriptions,
        "bag_path": str(Path(bag_path).resolve()),
        "window_start": _utc_isoformat(initial_start),
        "window_end": _utc_isoformat(initial_end),
        "selection_policy": "explicit_fixed_bag",
        "timestamp_policy": "intersection",
    }

    if n_updates > 0 and save_checkpoint:
        model.save(checkpoint_path, checkpoint_type="coinbase_week_http")
        print(f"[Coinbase week/live] saved checkpoint {checkpoint_path}")
    if training_db_path:
        print("[Coinbase week/live] fixed-bag path is adaptation-only; winner publication disabled")

    if avg_loss is not None:
        walk_text = (
            f"{walk_forward_loss:.6f} updates={walk_forward_updates}"
            if walk_forward_loss is not None
            else "n/a"
        )
        print(
            f"[Coinbase week/live] initial train_avg_loss={avg_loss:.6f} "
            f"n_updates={n_updates} walk_forward={walk_text}"
        )
    else:
        print("[Coinbase week/live] initial training produced no scored updates")

    if not live:
        return

    coinbase_pairs = [sub["product_id"] for sub in selected_subscriptions if sub["exchange"] == "coinbase"]
    if not coinbase_pairs:
        raise NotImplementedError("Live streaming currently requires at least one coinbase subscription")

    stop_event = threading.Event()
    ingest_thread = threading.Thread(
        target=cache.stream_live,
        kwargs={
            "pairs": coinbase_pairs,
            "granularity": granularity,
            "stop_event": stop_event,
            "repair_every_seconds": live_http_repair_seconds,
            "overlap_candles": live_http_overlap_candles,
            "idle_restart_seconds": live_idle_restart_seconds,
            "exchange": "coinbase",
        },
        daemon=True,
    )
    ingest_thread.start()

    last_trained_bars = len(graph.common_timestamps)
    last_trained_timestamp = graph.common_timestamps[-1]

    try:
        while True:
            time.sleep(live_train_interval)
            now = _floor_time_to_granularity(_utc_now_naive(), granularity)
            purged = cache.purge_future_candles(granularity=granularity)
            if purged:
                print(f"[Coinbase live] purged {purged} poisoned future candles")
            window_start = now - timedelta(days=lookback_days)
            graph = _load_selected_pair_graph(
                selected_subscriptions,
                window_start,
                now,
                granularity=granularity,
            )
            total_bars = len(graph.common_timestamps)
            if total_bars == 0:
                print("[Coinbase live] no bars available after reload")
                continue

            latest_timestamp = graph.common_timestamps[-1]
            if latest_timestamp <= last_trained_timestamp:
                continue

            model.register_edges(list(graph.edges.keys()))
            start_bar = max(0, total_bars - _replay_margin_bars(model))
            replay_split = plan_walk_forward_split(total_bars - start_bar, model)
            if replay_split is None:
                print("[Coinbase live] skipping cycle: not enough replay bars for walk-forward split")
                last_trained_bars = total_bars
                last_trained_timestamp = latest_timestamp
                continue
            _, train_end_rel, validation_end_rel = replay_split
            train_end_bar = start_bar + train_end_rel
            validation_end_bar = start_bar + validation_end_rel
            print(
                f"[Coinbase live] retraining bars {start_bar}:{train_end_bar} "
                f"and validating {train_end_bar}:{validation_end_bar} "
                f"({total_bars - start_bar} replay bars, last_trained={last_trained_bars}, "
                f"latest_ts={latest_timestamp})"
            )
            total_loss, n_updates, _, _ = run_training(
                graph,
                model,
                start_bar=start_bar,
                end_bar=train_end_bar,
                print_every=print_every,
            )
            if n_updates <= 0:
                last_trained_bars = total_bars
                last_trained_timestamp = latest_timestamp
                print("[Coinbase live] no scored updates in this cycle")
                continue

            avg_loss = total_loss / n_updates
            walk_forward_loss, walk_forward_updates = run_walk_forward_validation(
                graph,
                model,
                validation_start_bar=train_end_bar,
                end_bar=validation_end_bar,
                warmup_start_bar=start_bar,
            )
            print(
                f"[Coinbase live] model_cas={model.model_cas_signature()} "
                f"rotary={_rotary_state_summary(model)}"
            )
            cycle_params = {
                **week_params,
                "h_dim": model.h_dim,
                "z_dim": model.z_dim,
                "cycle_window_start": _utc_isoformat(window_start),
                "cycle_window_end": _utc_isoformat(now),
                "start_bar": start_bar,
                "train_end_bar": train_end_bar,
                "validation_end_bar": validation_end_bar,
            }
            cycle_bag_spec = {
                "pairs": len(selected_subscriptions),
                "subscriptions": selected_subscriptions,
                "bag_path": str(Path(bag_path).resolve()),
                "window_start": _utc_isoformat(window_start),
                "window_end": _utc_isoformat(now),
                "total_bars": total_bars,
                "selection_policy": "explicit_fixed_bag",
                "timestamp_policy": "intersection",
            }

            if save_checkpoint or (
                walk_forward_loss is not None and walk_forward_loss < best_loss
            ):
                model.save(checkpoint_path, checkpoint_type="coinbase_live")
                print(f"[Coinbase live] saved checkpoint {checkpoint_path}")

            if walk_forward_loss is not None and walk_forward_loss < best_loss:
                best_loss = walk_forward_loss
                print(
                    f"[Coinbase live] new best walk_forward_loss={walk_forward_loss:.6f} "
                    f"(train_avg_loss={avg_loss:.6f}, updates={walk_forward_updates})"
                )
                if training_db_path:
                    print(
                        "[Coinbase live] fixed-bag path is adaptation-only; "
                        "winner publication disabled"
                    )
            else:
                best_text = f"{best_loss:.6f}" if np.isfinite(best_loss) else "n/a"
                walk_text = (
                    f"{walk_forward_loss:.6f}"
                    if walk_forward_loss is not None
                    else "n/a"
                )
                print(
                    f"[Coinbase live] train_avg_loss={avg_loss:.6f} "
                    f"walk_forward_loss={walk_text} best={best_text}"
                )

            last_trained_bars = total_bars
            last_trained_timestamp = latest_timestamp
    except KeyboardInterrupt:
        print("[Coinbase live] stopping")
    finally:
        stop_event.set()
        ingest_thread.join(timeout=5.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoresearch', action='store_true')
    parser.add_argument('--coinbase-week-http-then-live', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--start-bar', type=int, default=0)
    parser.add_argument('--end-bar', type=int, default=None)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--min-partners', type=int, default=5)
    parser.add_argument('--max-partners', type=int, default=None)
    parser.add_argument('--skip-fetch', action='store_true')
    parser.add_argument('--bag', type=str, default=str(Config.BAG_PATH))
    parser.add_argument('--lookback-days', type=int, default=7)
    parser.add_argument('--granularity', type=str, default="300")
    parser.add_argument('--exchange', type=str, default='coinbase', choices=['coinbase', 'binance'])
    parser.add_argument('--prediction-depth', type=int, default=1)
    parser.add_argument('--h-dim', type=int, default=4)
    parser.add_argument('--z-dim', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--y-depth', type=int, default=200)
    parser.add_argument('--x-pixels', type=int, default=20)
    parser.add_argument('--curvature', type=float, default=2.0)
    parser.add_argument('--checkpoint-path', type=str, default='model_weights.pt')
    parser.add_argument('--training-db-path', type=str, default=None)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--live-train-interval', type=int, default=60)
    parser.add_argument('--live-http-repair-seconds', type=int, default=30)
    parser.add_argument('--live-http-overlap-candles', type=int, default=6)
    parser.add_argument('--live-idle-restart-seconds', type=int, default=90)
    args = parser.parse_args()

    if args.coinbase_week_http_then_live:
        run_coinbase_week_http_then_live(
            bag_path=args.bag,
            checkpoint_path=args.checkpoint_path,
            training_db_path=args.training_db_path,
            lookback_days=args.lookback_days,
            granularity=args.granularity,
            learning_rate=args.lr,
            y_depth=args.y_depth,
            x_pixels=args.x_pixels,
            curvature=args.curvature,
            prediction_depth=args.prediction_depth,
            h_dim=args.h_dim,
            z_dim=args.z_dim,
            device=args.device,
            print_every=args.print_every,
            save_checkpoint=args.save_checkpoint,
            live=args.live,
            live_train_interval=args.live_train_interval,
            live_http_repair_seconds=args.live_http_repair_seconds,
            live_http_overlap_candles=args.live_http_overlap_candles,
            live_idle_restart_seconds=args.live_idle_restart_seconds,
        )
        return

    ensure_pool_running(str(Config.DB_PATH))
    print(f"[DuckDB pool] ready db={Path(Config.DB_PATH).resolve()}")

    print("Loading coin graph...")
    graph = CoinGraph(fee_rate=0.001)
    n_bars = graph.load(
        lookback_days=1095 if args.exchange == "binance" else 365,
        min_partners=args.min_partners,
        max_partners=args.max_partners,
        exchange=args.exchange,
        skip_fetch=args.skip_fetch,
    )
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges, {n_bars} bars")
    
    if n_bars == 0:
        print("No data. Run fetch_candles.py first.")
        return

    model = HierarchicalReasoningModel(
        n_edges=len(graph.edges),
        learning_rate=args.lr,
        y_depth=args.y_depth,
        x_pixels=args.x_pixels,
        curvature=args.curvature,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        prediction_depth=args.prediction_depth,
        device=args.device,
    )
    model.register_edges(list(graph.edges.keys()))
    model.load(args.checkpoint_path)

    db_path = str(Config.DB_PATH)

    if args.autoresearch:
        run_autoresearch(
            graph,
            db_path=db_path,
            exchange=args.exchange,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            training_db_path=args.training_db_path,
        )
    else:
        # Stochastic bag sampling for single training run (Binance mode)
        if args.exchange == "binance":
            all_db_subscriptions = _list_all_exchange_subscriptions(db_path, exchange=args.exchange)
            if all_db_subscriptions:
                filtered = _compute_volatility_filter(
                    db_path, all_db_subscriptions,
                    lookback_days=1095, granularity="300", min_velocity=0.001,
                )
                print(
                    f"[StochasticBag] Volatility-filtered: {len(filtered)} subscriptions "
                    f"(from {len(all_db_subscriptions)})"
                )
                if filtered:
                    rng = random.Random()
                    sampled = _stochastic_bag_sample(
                        [sub["product_id"] for sub in filtered], args.h_dim, rng,
                        min_pairs=5, max_pairs=len(filtered),
                    )
                    if sampled:
                        print(f"[StochasticBag] Sampled {len(sampled)} pairs for training")
                        # Rebuild graph with sampled pairs only
                        graph = CoinGraph(fee_rate=0.001)
                        graph.load(
                            lookback_days=1095,
                            min_partners=args.min_partners,
                            max_partners=args.max_partners,
                            exchange=args.exchange,
                            skip_fetch=True,
                        )
                        # Filter graph edges to sampled pairs only
                        pair_to_edges = {}
                        for sub in getattr(graph, "bag_subscriptions", []):
                            pid = sub["product_id"]
                            sub_exchange = sub["exchange"]
                            parts = pid.split("-", 1)
                            if len(parts) == 2:
                                pair_to_edges[pid] = (sub_exchange, parts[0], parts[1])
                        from coin_graph import EdgeState, NodeState
                        new_edges = {}
                        new_edge_state = {}
                        new_nodes = set()
                        new_node_state = {}
                        for pid in sampled:
                            if pid not in pair_to_edges:
                                continue
                            edge_exchange, base, quote = pair_to_edges[pid]
                            for edge in [(edge_exchange, base, quote), (edge_exchange, quote, base)]:
                                if edge in graph.edges:
                                    new_edges[edge] = graph.edges[edge]
                                    new_edge_state[edge] = EdgeState()
                            new_nodes.add(base)
                            new_nodes.add(quote)
                            new_node_state.setdefault(base, NodeState())
                            new_node_state.setdefault(quote, NodeState())
                        if new_edges:
                            graph.edges = new_edges
                            graph.edge_state = new_edge_state
                            graph.nodes = new_nodes
                            graph.node_state = new_node_state
                            graph.bag_subscriptions = [{"exchange": args.exchange, "product_id": pid} for pid in sampled]
                            graph.all_pairs = sampled
                            graph._align_timestamps()
                            n_bars = len(graph.common_timestamps)
                            # Re-register model edges
                            model = HierarchicalReasoningModel(
                                n_edges=len(graph.edges),
                                learning_rate=args.lr,
                                y_depth=args.y_depth,
                                x_pixels=args.x_pixels,
                                curvature=args.curvature,
                                h_dim=args.h_dim,
                                z_dim=args.z_dim,
                                prediction_depth=args.prediction_depth,
                                device=args.device,
                            )
                            model.register_edges(list(graph.edges.keys()))
                            model.load(args.checkpoint_path)

        loss_history = []
        profile_stats = {} if args.profile else None
        end_bar = args.end_bar if args.end_bar else min(n_bars, 10000)
        print(f"Training from bar {args.start_bar} to {end_bar}...")

        total_loss, n_updates, _, loss_history = run_training(
            graph, model,
            start_bar=args.start_bar,
            end_bar=end_bar,
            print_every=args.print_every,
            loss_history=loss_history,
            profile_stats=profile_stats,
        )

        print(f"\nDone: {_training_status(total_loss, n_updates)}")
        if profile_stats is not None:
            _print_profile_summary(profile_stats)
        if args.save_checkpoint and n_updates > 0:
            model.save(args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")
        elif args.save_checkpoint:
            print(f"Skipped checkpoint save to {args.checkpoint_path}: no scored updates")
        else:
            print("Checkpoint not saved; pass --save-checkpoint for an intentional training run")


if __name__ == "__main__":
    main()
