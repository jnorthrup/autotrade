"""
showdown.py - Unified HRM Training, Orchestration, and Dashboard CLI.
Consolidates graph_showdown.py, finetune.py, training_worker.py, training_orchestrator.py, dashboard_showdown.py, and health_monitor.py.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import duckdb
import zstandard as zstd
import schedule
from flask import Flask, jsonify, render_template_string

from cache import (
    Config,
    CandleCache,
    CoinGraph,
    EdgeState,
    NodeState,
    PoolClient,
    ensure_pool_running,
    pool_is_running,
    _utc_now_naive,
    _utc_isoformat,
    _floor_time_to_granularity,
)
from model import HierarchicalReasoningModel
from wallet import OrderShim, SimWallet

# --- Constants & Helpers ---
SQUARE_CUBE_SIZES = [4, 16, 64, 256]
PLATEAU_WINDOW = 100
PLATEAU_THRESHOLD = 1e-5
PLATEAU_PATIENCE = 3
WALK_FORWARD_VALIDATION_FRACTION = 0.2


def _choose_value_asset(graph: CoinGraph) -> str:
    for asset in ("USD", "USDT", "USDC", "BTC", "ETH"):
        if asset in getattr(graph, "nodes", set()):
            return asset
    nodes = sorted(getattr(graph, "nodes", set()))
    return nodes[0] if nodes else "USD"


def _init_wallet(graph: CoinGraph, capital: float) -> SimWallet:
    value_asset = _choose_value_asset(graph)
    initial_balances = {asset: 0.0 for asset in getattr(graph, "nodes", set())}
    initial_balances[value_asset] = float(capital)
    return SimWallet(
        getattr(graph, "nodes", set()),
        value_asset=value_asset,
        initial_balances=initial_balances,
    )


def _orders_from_predictions(
    wallet: SimWallet,
    graph: CoinGraph,
    predictions: Dict[Tuple[str, str, str], Tuple[float, float, float]],
    *,
    bar_idx: int,
    prediction_depth: int,
    threshold: float = 0.55,
) -> List[OrderShim]:
    if not predictions:
        return []

    free_before = wallet.free_qty_map()
    requested_by_asset: Dict[str, float] = {}
    prepared: List[Tuple[Tuple[str, str, str], bool, str, float, float, float]] = []

    for edge, raw in predictions.items():
        frac, ptt, stop = (float(raw[0]), float(raw[1]), float(raw[2]))
        frac = min(1.0, max(0.0, frac))
        if frac <= 0.0:
            continue
        is_buy = ptt > threshold and ptt >= stop
        is_sell = stop > threshold and stop > ptt
        if not is_buy and not is_sell:
            continue
        price = wallet.edge_price(graph, edge, bar_idx)
        if price <= 0.0:
            continue
        _, base_asset, quote_asset = edge
        spend_asset = quote_asset if is_buy else base_asset
        requested_by_asset[spend_asset] = requested_by_asset.get(spend_asset, 0.0) + frac
        prepared.append((edge, is_buy, spend_asset, frac, max(ptt, stop), price))

    orders: List[OrderShim] = []
    for edge, is_buy, spend_asset, frac, confidence, price in prepared:
        total_frac = requested_by_asset.get(spend_asset, 0.0)
        scaled_frac = frac / total_frac if total_frac > 1.0 else frac
        spend_qty = max(0.0, float(free_before.get(spend_asset, 0.0)) * scaled_frac)
        if spend_qty <= 0.0:
            continue
        orders.append(
            OrderShim(
                edge=edge,
                price=price,
                is_buy=is_buy,
                amt=spend_qty,
                created_bar=bar_idx,
                maturity_bar=bar_idx + max(1, int(prediction_depth)),
                spend_asset=spend_asset,
                confidence=confidence,
                fraction=scaled_frac,
            )
        )

    return orders


def _canonical_pair_edges(graph: CoinGraph) -> List[Tuple[str, str, str]]:
    canonical: List[Tuple[str, str, str]] = []
    seen_products = set()

    for bag_item in getattr(graph, "all_pairs", []) or []:
        if ":" not in str(bag_item):
            continue
        exchange, product_id = str(bag_item).split(":", 1)
        product_id = product_id.strip()
        if product_id in seen_products or "-" not in product_id:
            continue
        base_asset, quote_asset = product_id.split("-", 1)
        canonical.append((exchange.strip(), base_asset, quote_asset))
        seen_products.add(product_id)

    if canonical:
        return canonical

    edge_product_id = getattr(graph, "edge_product_id", {})
    for edge in getattr(graph, "edges", {}).keys():
        product_id = str(edge_product_id.get(edge, "")).strip()
        if product_id in seen_products or "-" not in product_id:
            continue
        base_asset, quote_asset = product_id.split("-", 1)
        canonical.append((str(edge[0]), base_asset, quote_asset))
        seen_products.add(product_id)

    return canonical if canonical else list(getattr(graph, "edges", {}).keys())


def _use_pool_for_db(db_path: str) -> bool:
    try:
        return (
            pool_is_running()
            and Path(db_path or Config.DB_PATH).resolve()
            == Path(Config.DB_PATH).resolve()
        )
    except Exception:
        return False


def _pool() -> PoolClient:
    return PoolClient()


def _next_square_cube_size(size: int) -> Optional[int]:
    if size == 1:
        return SQUARE_CUBE_SIZES[0]
    if size < 1 or size % 4 != 0:
        return None
    return size * 4


def _growable_growth_dims(options: Tuple[Tuple[str, int], ...]) -> List[str]:
    return [dim for dim, size in options if _next_square_cube_size(size) is not None]


def _apply_growth_step(
    model: HierarchicalReasoningModel,
    growth_dim: str,
    hidden_size: int,
    H_layers: int,
    L_layers: int,
    H_cycles: int = 1,
    L_cycles: int = 1,
) -> Tuple[int, int, int, int, int, bool]:
    if growth_dim == "h":
        if _next_square_cube_size(hidden_size) is None:
            return hidden_size, H_layers, L_layers, H_cycles, L_cycles, False
        model.grow("h")
        hidden_size = model.h_dim
    elif growth_dim == "H":
        if _next_square_cube_size(H_layers) is None:
            return hidden_size, H_layers, L_layers, H_cycles, L_cycles, False
        model.grow("H")
        H_layers = model.H_layers
    elif growth_dim == "L":
        if _next_square_cube_size(L_layers) is None:
            return hidden_size, H_layers, L_layers, H_cycles, L_cycles, False
        model.grow("L")
        L_layers = model.L_layers
    elif growth_dim == "Hc":
        if _next_square_cube_size(H_cycles) is None:
            return hidden_size, H_layers, L_layers, H_cycles, L_cycles, False
        model.grow("Hc")
        H_cycles = model.H_cycles
    elif growth_dim == "Lc":
        if _next_square_cube_size(L_cycles) is None:
            return hidden_size, H_layers, L_layers, H_cycles, L_cycles, False
        model.grow("Lc")
        L_cycles = model.L_cycles
    else:
        raise ValueError(f"Unknown growth dimension: {growth_dim}")
    return hidden_size, H_layers, L_layers, H_cycles, L_cycles, True


def _is_converged(losses: List[float]) -> bool:
    if len(losses) < PLATEAU_WINDOW * PLATEAU_PATIENCE:
        return False
    for i in range(PLATEAU_PATIENCE):
        chunk = losses[-PLATEAU_WINDOW * (PLATEAU_PATIENCE - i) :]
        if len(chunk) < PLATEAU_WINDOW:
            continue
        if (
            abs(np.mean(chunk[-PLATEAU_WINDOW:]) - np.mean(chunk[:PLATEAU_WINDOW]))
            > PLATEAU_THRESHOLD
        ):
            return False
    return True


def _training_status(total_loss: float, n_updates: int) -> str:
    return (
        f"avg_loss={total_loss / n_updates:.6f}, n_updates={n_updates}"
        if n_updates > 0
        else "no scored updates; warmup or maturity never completed"
    )


def plan_walk_forward_split(
    total_bars: int,
    model: HierarchicalReasoningModel,
    validation_fraction: float = WALK_FORWARD_VALIDATION_FRACTION,
) -> Optional[Tuple[int, int, int]]:
    min_train_bars = max(model.y_depth + model.prediction_depth + 64, 256)
    min_validation_bars = max(model.prediction_depth + 32, 64)
    if total_bars < (min_train_bars + min_validation_bars):
        return None
    validation_bars = min(
        max(min_validation_bars, int(total_bars * validation_fraction)),
        total_bars - min_train_bars,
    )
    if (
        validation_bars < min_validation_bars
        or total_bars - validation_bars < min_train_bars
    ):
        return None
    return 0, total_bars - validation_bars, total_bars


def _profit_based_loss(
    wallet: SimWallet,
    graph: CoinGraph,
    *,
    bar_idx: int,
    bar_pnl: float,
    capital: float,
    max_pnl: float,
    pnl_history: List[float],
    stagnation_window: int = 20,
    inactivity_penalty: float = 0.001,
    stagnation_penalty: float = 0.0005,
) -> Tuple[float, float, float, float]:
    current_balance = wallet.worth(graph, bar_idx)
    total_pnl = current_balance - capital
    max_pnl = max(max_pnl, total_pnl)
    pnl_history.append(bar_pnl)
    loss = abs(bar_pnl) / max(capital, 1.0) if bar_pnl < 0 else 0.0
    if wallet.open_order_count == 0:
        loss += inactivity_penalty
    if len(pnl_history) >= stagnation_window:
        recent = np.mean(pnl_history[-stagnation_window:])
        older = (
            np.mean(pnl_history[-2 * stagnation_window : -stagnation_window])
            if len(pnl_history) >= 2 * stagnation_window
            else recent
        )
        if recent <= older:
            loss += stagnation_penalty * (
                1
                + max(
                    0,
                    stagnation_window
                    - len([p for p in pnl_history[-stagnation_window:] if p > 0]),
                )
            )
    if max_pnl - total_pnl > 0:
        loss += (max_pnl - total_pnl) / max(capital, 1.0) * 0.01
    return loss, bar_pnl, total_pnl, max_pnl


def run_walk_forward_validation(
    graph: CoinGraph,
    trained_model: HierarchicalReasoningModel,
    validation_start_bar: int,
    end_bar: Optional[int] = None,
    warmup_start_bar: int = 0,
    print_every: int = 0,
    capital: float = 100.0,
    use_profit_loss: bool = False,
) -> Tuple[Optional[float], int]:
    if end_bar is None:
        end_bar = len(graph.common_timestamps)
    if validation_start_bar >= end_bar:
        return None, 0
    eval_graph = CoinGraph(
        fee_rate=graph.fee_rate, min_pair_coverage=graph.min_pair_coverage
    )
    (
        eval_graph.all_pairs,
        eval_graph.bag_subscriptions,
        eval_graph.nodes,
        eval_graph.common_timestamps,
    ) = (
        list(graph.all_pairs),
        list(getattr(graph, "bag_subscriptions", [])),
        set(graph.nodes),
        list(graph.common_timestamps),
    )
    for edge, df in graph.edges.items():
        eval_graph.edges[edge], eval_graph.edge_state[edge] = df, EdgeState()
    eval_graph.edge_product_id, eval_graph.edge_is_inverted = (
        dict(getattr(graph, "edge_product_id", {})),
        dict(getattr(graph, "edge_is_inverted", {})),
    )
    for node in graph.node_state:
        eval_graph.node_state[node] = NodeState()
    eval_model = HierarchicalReasoningModel(
        n_edges=len(graph.edges.keys()),
        learning_rate=trained_model._lr,
        y_depth=trained_model.y_depth,
        x_pixels=trained_model.x_pixels,
        curvature=trained_model.curvature,
        h_dim=trained_model.h_dim,
        z_dim=trained_model.z_dim,
        prediction_depth=trained_model.prediction_depth,
        H_layers=trained_model.H_layers,
        L_layers=trained_model.L_layers,
        H_cycles=trained_model.H_cycles,
        L_cycles=trained_model.L_cycles,
        device=trained_model.device_preference,
    )
    print("  About to register edges", flush=True)
    eval_model.register_edges(_canonical_pair_edges(graph))
    if trained_model._model and eval_model._model:
        eval_model._model.load_state_dict(trained_model._model.state_dict())
    wallet = _init_wallet(eval_graph, capital)

    (
        total_loss,
        n_updates,
        active_predictions,
        total_pnl,
        max_pnl,
        max_drawdown,
        pnl_history,
    ) = 0.0, 0, {}, 0.0, 0.0, 0.0, []
    for bar_idx in range(warmup_start_bar, end_bar):
        edge_accels, edge_velocities, hit_ptt, hit_stop = eval_graph.update(bar_idx)
        if not edge_accels:
            continue
        eval_model.update_prices(eval_graph, bar_idx)
        if use_profit_loss:
            settlement = wallet.settle_due_orders(
                eval_graph, bar_idx=bar_idx, fee_rate=eval_graph.fee_rate
            )
            if eval_model.ready_for_prediction(bar_idx):
                wallet.reserve_orders(
                    _orders_from_predictions(
                        wallet,
                        eval_graph,
                        eval_model.predict(eval_graph, bar_idx),
                        bar_idx=bar_idx,
                        prediction_depth=eval_model.prediction_depth,
                    ),
                    graph=eval_graph,
                    bar_idx=bar_idx,
                )
            loss, bar_pnl, total_pnl, max_pnl = _profit_based_loss(
                wallet,
                eval_graph,
                bar_idx=bar_idx,
                bar_pnl=settlement["bar_pnl"],
                capital=capital,
                max_pnl=max_pnl,
                pnl_history=pnl_history,
            )
            if loss is not None and bar_idx >= validation_start_bar:
                total_loss += loss
                n_updates += 1
        else:
            if eval_model.ready_for_prediction(bar_idx):
                active_predictions[bar_idx] = eval_model.predict(eval_graph, bar_idx)
            if not eval_model.ready_for_update(bar_idx, edge_accels):
                continue
            loss = eval_model.score(
                eval_graph,
                edge_accels,
                bar_idx,
                actual_velocities=edge_velocities,
                hit_ptt=hit_ptt,
                hit_stop=hit_stop,
            )
            mature_idx = bar_idx - eval_model.prediction_depth
            if mature_idx in active_predictions:
                bar_pnl = sum(
                    (capital + total_pnl) * frac * (vel - eval_graph.fee_rate)
                    if ptt > 0.55
                    else (capital + total_pnl) * frac * (-vel - eval_graph.fee_rate)
                    if stop > 0.55
                    else 0.0
                    for edge, (frac, ptt, stop) in active_predictions.pop(
                        mature_idx
                    ).items()
                    if edge in edge_velocities
                    for vel in [edge_velocities[edge]]
                )
                total_pnl += bar_pnl
                max_pnl = max(max_pnl, total_pnl)
                max_drawdown = max(max_drawdown, max_pnl - total_pnl)
            if loss is not None and bar_idx >= validation_start_bar:
                total_loss += loss
                n_updates += 1
    return (total_loss / n_updates, n_updates) if n_updates > 0 else (None, 0)


def run_training(
    graph: CoinGraph,
    model: HierarchicalReasoningModel,
    start_bar: int = 0,
    end_bar: Optional[int] = None,
    print_every: int = 100,
    loss_history: Optional[List[float]] = None,
    profile_stats: Optional[Dict[str, float]] = None,
    capital: float = 100.0,
    use_profit_loss: bool = False,
) -> Tuple[float, int, bool, List[float]]:
    if end_bar is None:
        end_bar = len(graph.common_timestamps)
    loss_history = loss_history or []
    print("  - set profile")
    model.set_profile_enabled(profile_stats is not None)
    print("  - calc ts_to_bar")
    ts_to_bar = {ts: i for i, ts in enumerate(graph.common_timestamps)}
    bars_with_data = set()
    print("  - for df in graph.edges")
    for df in graph.edges.values():
        bars_with_data.update(ts_to_bar[ts] for ts in set(df.index) & ts_to_bar.keys())
    (
        total_loss,
        n_updates,
        active_predictions,
        total_pnl,
        max_pnl,
        max_drawdown,
        pnl_history,
    ) = 0.0, 0, {}, 0.0, 0.0, 0.0, []
    wallet = _init_wallet(graph, capital)
    print("  - sorted_bars")
    sorted_bars = sorted(b for b in bars_with_data if start_bar <= b < end_bar)
    print(f"Training on {len(sorted_bars)} bars with data")
    for i, bar_idx in enumerate(sorted_bars):
        if bar_idx >= len(graph.common_timestamps):
            break
        edge_accels, edge_velocities, hit_ptt, hit_stop = graph.update(bar_idx)
        if not edge_accels:
            continue
        model.update_prices(graph, bar_idx)
        if use_profit_loss:
            settlement = wallet.settle_due_orders(
                graph, bar_idx=bar_idx, fee_rate=graph.fee_rate
            )
            if model.ready_for_prediction(bar_idx):
                wallet.reserve_orders(
                    _orders_from_predictions(
                        wallet,
                        graph,
                        model.predict(graph, bar_idx),
                        bar_idx=bar_idx,
                        prediction_depth=model.prediction_depth,
                    ),
                    graph=graph,
                    bar_idx=bar_idx,
                )
            # Wallet PnL is useful telemetry, but it is not differentiable.
            # Take a real optimizer step with the model's training loss so the
            # predictions can actually move and eventually produce trades.
            train_loss = None
            if model.ready_for_update(bar_idx, edge_accels, graph=graph):
                train_loss = model.update(
                    graph,
                    edge_accels,
                    bar_idx,
                    actual_velocities=edge_velocities,
                    hit_ptt=hit_ptt,
                    hit_stop=hit_stop,
                )
            _profit_loss, bar_pnl, total_pnl, max_pnl = _profit_based_loss(
                wallet,
                graph,
                bar_idx=bar_idx,
                bar_pnl=settlement["bar_pnl"],
                capital=capital,
                max_pnl=max_pnl,
                pnl_history=pnl_history,
            )
            loss = train_loss
            if loss is not None:
                total_loss += loss
                n_updates += 1
                loss_history.append(loss)
        else:
            if model.ready_for_prediction(bar_idx):
                active_predictions[bar_idx] = model.predict(graph, bar_idx)
            if model.ready_for_update(bar_idx, edge_accels, graph=graph):
                loss = model.update(
                    graph,
                    edge_accels,
                    bar_idx,
                    actual_velocities=edge_velocities,
                    hit_ptt=hit_ptt,
                    hit_stop=hit_stop,
                )
                if loss is not None:
                    total_loss += loss
                    n_updates += 1
                    loss_history.append(loss)
            mature_idx = bar_idx - model.prediction_depth
            if mature_idx in active_predictions:
                bar_pnl = sum(
                    (capital + total_pnl) * frac * (vel - graph.fee_rate)
                    if ptt > 0.55
                    else (capital + total_pnl) * frac * (-vel - graph.fee_rate)
                    if stop > 0.55
                    else 0.0
                    for edge, (frac, ptt, stop) in active_predictions.pop(
                        mature_idx
                    ).items()
                    if edge in edge_velocities
                    for vel in [edge_velocities[edge]]
                )
                total_pnl += bar_pnl
                max_pnl = max(max_pnl, total_pnl)
                max_drawdown = max(max_drawdown, max_pnl - total_pnl)
        if (i % print_every == 0 and i > 0) or (i == len(sorted_bars) - 1):
            if n_updates > 0:
                balance = (
                    wallet.worth(graph, bar_idx)
                    if use_profit_loss
                    else capital + total_pnl
                )
                if use_profit_loss:
                    max_drawdown = max(max_drawdown, max_pnl - total_pnl)
                print(
                    f"Train[{i + 1}/{len(sorted_bars)}] Bar {bar_idx}: avg_loss={total_loss / n_updates:.6f}, updates={n_updates}, pnl={total_pnl:.5f}, balance={balance:.5f}, max_dd={max_drawdown:.5f}",
                    flush=True,
                )
            else:
                print(
                    f"Train[{i + 1}/{len(sorted_bars)}] Bar {bar_idx}: warmup",
                    flush=True,
                )
    if profile_stats is not None:
        profile_stats.clear()
        profile_stats.update(model.get_profile_stats())
    return total_loss, n_updates, False, loss_history


# --- Autoresearch ---
def _list_all_exchange_subscriptions(
    db_path: str, exchange: Optional[str] = None
) -> List[Dict[str, str]]:
    try:
        if _use_pool_for_db(db_path):
            rows = _pool().execute(
                "SELECT DISTINCT exchange, product_id FROM candles"
                if exchange is None
                else "SELECT DISTINCT exchange, product_id FROM candles WHERE exchange = ?",
                [] if exchange is None else [exchange],
            )
        else:
            with duckdb.connect(db_path, read_only=True) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT exchange, product_id FROM candles"
                    if exchange is None
                    else "SELECT DISTINCT exchange, product_id FROM candles WHERE exchange = ?",
                    [] if exchange is None else [exchange],
                ).fetchall()
        deduped = {}
        for row in rows:
            if isinstance(row, (tuple, list)) and len(row) >= 2:
                ex, pid = str(row[0]).strip(), str(row[1]).strip()
                if ex and "-" in pid:
                    deduped[(ex, pid)] = {"exchange": ex, "product_id": pid}
        return list(deduped.values())
    except Exception:
        return []


def _compute_volatility_filter(
    db_path: str,
    all_pairs: List[Dict[str, str]],
    lookback_days: int = 365,
    granularity: str = "300",
    min_velocity: float = 0.001,
) -> List[Dict[str, str]]:
    end = _utc_now_naive()
    start = end - timedelta(days=lookback_days)
    filtered = []
    use_pool_flag = _use_pool_for_db(db_path)
    pool = _pool() if use_pool_flag else None
    for sub in all_pairs:
        pid, exchange = sub["product_id"], sub["exchange"]
        parts = pid.split("-", 1)
        if len(parts) != 2 or (
            parts[0] in {"USD", "USDT", "EUR"} and parts[1] in {"USD", "USDT", "EUR"}
        ):
            continue
        try:
            if use_pool_flag:
                rows = pool.execute(
                    "SELECT AVG(ABS(LN(close / NULLIF(open, 0)))) FROM candles WHERE exchange = ? AND product_id = ? AND granularity = ? AND timestamp >= ? AND timestamp < ? AND open > 0 AND close > 0",
                    [exchange, pid, granularity, start, end],
                )
                mean_vel = (
                    rows[0][0] if rows and rows[0] and rows[0][0] is not None else 0.0
                )
            else:
                with duckdb.connect(db_path, read_only=True) as conn:
                    row = conn.execute(
                        "SELECT AVG(ABS(LN(close / NULLIF(open, 0)))) FROM candles WHERE exchange = ? AND product_id = ? AND granularity = ? AND timestamp >= ? AND timestamp < ? AND open > 0 AND close > 0",
                        [exchange, pid, granularity, start, end],
                    ).fetchone()
                    mean_vel = row[0] if row and row[0] is not None else 0.0
            if mean_vel >= min_velocity:
                filtered.append(sub)
        except Exception:
            pass
    return filtered


def _stochastic_bag_sample(
    filtered_pairs: List[str],
    model_size: int,
    rng: random.Random,
    min_pairs: int = 5,
    max_pairs: Optional[int] = None,
) -> List[str]:
    if not filtered_pairs:
        return []
    target = {4: 5, 16: 20, 64: 40, 256: 80}.get(
        model_size, max(min_pairs, model_size * 3 // 4)
    )
    if max_pairs is not None:
        target = min(target, max_pairs)
    target = max(min_pairs, min(target, len(filtered_pairs)))
    adj = {}
    for pid in filtered_pairs:
        parts = pid.split("-", 1)
        if len(parts) == 2:
            adj.setdefault(parts[0], []).append(pid)
            adj.setdefault(parts[1], []).append(pid)
    if target >= len(filtered_pairs) or not adj:
        return list(filtered_pairs)[:target]
    selected, visited_currencies, frontier = (
        set(),
        {rng.choice(list(adj.keys()))},
        [rng.choice(list(adj.keys()))],
    )
    while len(selected) < target and frontier:
        curr = frontier.pop(0)
        candidates = [p for p in adj.get(curr, []) if p not in selected]
        rng.shuffle(candidates)
        for pid in candidates:
            if len(selected) >= target:
                break
            selected.add(pid)
            for c in pid.split("-", 1):
                if c not in visited_currencies:
                    visited_currencies.add(c)
                    frontier.append(c)
        if not frontier and len(selected) < target:
            remaining = [c for c in adj.keys() if c not in visited_currencies]
            if remaining:
                new_seed = rng.choice(remaining)
                frontier.append(new_seed)
                visited_currencies.add(new_seed)
    return list(selected)


def run_autoresearch(
    db_path: str = "candles.duckdb",
    exchange: str = "binance",
    device: str = "auto",
    checkpoint_path: Optional[str] = None,
    training_db_path: Optional[str] = None,
    fixed_dim: Optional[int] = None,
):
    import cache

    cache.CandleCache(db_path)
    print("Using HRMEdgePredictor for hierarchical reasoning")
    experiments_db_path = str(Path(training_db_path or "training.duckdb"))
    if Path(experiments_db_path).resolve() == Path(db_path).resolve():
        print("[Autoresearch] WARNING: experiments DB shares the candle cache path")
    with duckdb.connect(experiments_db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS experiments (timestamp TIMESTAMP DEFAULT now(), val_bpb DOUBLE, params VARCHAR, bag_spec VARCHAR, growth_phase VARCHAR, model_cas VARCHAR)"
        )

    if _use_pool_for_db(db_path):
        rows = _pool().execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE exchange = ?",
            [exchange],
        )
    else:
        with duckdb.connect(db_path, read_only=True) as conn:
            rows = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE exchange = ?",
                [exchange],
            ).fetchall()

    db_min_ts = (
        pd.Timestamp(
            rows[0][0]
            if rows and rows[0][0]
            else _utc_now_naive() - timedelta(days=365)
        )
        .to_pydatetime()
        .replace(tzinfo=None)
    )
    db_max_ts = (
        pd.Timestamp(rows[0][1] if rows and rows[0][1] else _utc_now_naive())
        .to_pydatetime()
        .replace(tzinfo=None)
    )
    total_seconds = (db_max_ts - db_min_ts).total_seconds()
    total_bars = int(total_seconds / 300)

    all_db_subscriptions = _list_all_exchange_subscriptions(db_path, exchange=exchange)
    if not all_db_subscriptions and exchange == "binance":
        print(
            f"[Autoresearch] No Binance pairs found in {db_path}. Auto-absorbing MVP Binance Vision history..."
        )
        import binance_cache

        # Load a quick default MVP bag for autoresearch
        for pair in [
            "BTC-USDT",
            "ETH-USDT",
            "SOL-USDT",
            "XRP-USDT",
            "ADA-USDT",
            "DOGE-USDT",
            "BNB-USDT",
            "LTC-USDT",
            "DOT-USDT",
        ]:
            binance_cache.fetch_binance_vision(db_path, pair, "300")
        all_db_subscriptions = _list_all_exchange_subscriptions(
            db_path, exchange=exchange
        )

    if not all_db_subscriptions:
        print("[StochasticBag] ERROR: no pairs available")
        return
    filtered_subscriptions = _compute_volatility_filter(
        db_path,
        all_db_subscriptions,
        lookback_days=1095 if exchange == "binance" else 365,
        granularity="300",
        min_velocity=0.001,
    )
    filtered_pairs = [sub["product_id"] for sub in filtered_subscriptions]
    if not filtered_pairs:
        print("[StochasticBag] ERROR: no pairs after filtering")
        return

    rng = random.Random()
    growth_idx, hidden_size, H_layers, L_layers, H_cycles, L_cycles, phase = (
        0,
        SQUARE_CUBE_SIZES[0],
        1,
        1,
        1,
        1,
        0,
    )
    best_bpb = float("inf")

    if fixed_dim is not None:
        hidden_size = H_layers = L_layers = H_cycles = L_cycles = fixed_dim
    elif checkpoint_path and Path(checkpoint_path).exists():
        seed_model = HierarchicalReasoningModel(device=device)
        seed_model.load(checkpoint_path)
        hidden_size, H_layers, L_layers, H_cycles, L_cycles = (
            seed_model.h_dim,
            seed_model.H_layers,
            seed_model.L_layers,
            seed_model.H_cycles,
            seed_model.L_cycles,
        )

    print(
        f"\nAutoresearch: {len(filtered_pairs)} filtered pairs, {total_bars} bars\nSquare Cube: h={hidden_size}, H={H_layers}, L={L_layers}, Hc={H_cycles}, Lc={L_cycles}"
    )

    try:
        while True:
            phase += 1
            progress = min(1.0, phase / 200.0)
            selected_pairs = _stochastic_bag_sample(
                filtered_pairs,
                hidden_size,
                rng,
                min_pairs=5,
                max_pairs=len(filtered_pairs),
            )
            window_bars = min(
                rng.randint(
                    max(
                        500 if exchange == "binance" else 200,
                        total_bars // (5 if exchange == "binance" else 20),
                    ),
                    max(
                        max(
                            500 if exchange == "binance" else 200,
                            total_bars // (5 if exchange == "binance" else 20),
                        ),
                        int(
                            max(
                                500 if exchange == "binance" else 200,
                                total_bars // (5 if exchange == "binance" else 20),
                            )
                            + (
                                total_bars
                                - max(
                                    500 if exchange == "binance" else 200,
                                    total_bars // (5 if exchange == "binance" else 20),
                                )
                            )
                            * progress
                        ),
                    ),
                ),
                total_bars,
            )
            window_seconds = window_bars * 300
            start_offset = rng.uniform(0, max(0, total_seconds - window_seconds))
            start_time = db_min_ts + timedelta(seconds=start_offset)
            end_time = start_time + timedelta(seconds=window_seconds)

            trial_graph = CoinGraph(fee_rate=0.001)
            trial_graph.load(
                db_path=db_path,
                granularity="300",
                exchange=exchange,
                skip_fetch=True,
                start_time=start_time,
                end_time=end_time,
                explicit_bag=[
                    {"exchange": exchange, "product_id": pid} for pid in selected_pairs
                ],
            )
            if not trial_graph.edges:
                print(f"Phase {phase}: empty graph, skipping")
                continue

            lr, y_depth, x_pixels, curvature, prediction_depth = (
                10 ** random.uniform(-4, -1.5),
                random.choice([100, 200, 300, 400]),
                random.choice([10, 15, 20, 30]),
                random.uniform(0.5, 4.0),
                random.choice([1, 2, 3, 5, 10]),
            )
            print(
                f"\n=== Phase {phase} ===\n  Square: h={hidden_size}, H={H_layers}, L={L_layers}, Hc={H_cycles}, Lc={L_cycles}\n  Bag: {len(selected_pairs)} pairs, {window_bars} bars"
            )

            print("  About to init model", flush=True)
            model = HierarchicalReasoningModel(
                n_edges=len(trial_graph.edges),
                learning_rate=lr,
                y_depth=y_depth,
                x_pixels=x_pixels,
                curvature=curvature,
                h_dim=hidden_size,
                z_dim=hidden_size,
                prediction_depth=prediction_depth,
                H_layers=H_layers,
                L_layers=L_layers,
                H_cycles=H_cycles,
                L_cycles=L_cycles,
                device=device,
            )
            print("  About to register edges", flush=True)
            model.register_edges(_canonical_pair_edges(trial_graph))
            if checkpoint_path and Path(checkpoint_path).exists() and fixed_dim is None:
                model.load(checkpoint_path)
                hidden_size, H_layers, L_layers, H_cycles, L_cycles = (
                    model.h_dim,
                    model.H_layers,
                    model.L_layers,
                    model.H_cycles,
                    model.L_cycles,
                )

            growth_options = _growable_growth_dims(
                (
                    ("h", hidden_size),
                    ("H", H_layers),
                    ("L", L_layers),
                    ("Hc", H_cycles),
                    ("Lc", L_cycles),
                )
            )
            growth_dim = rng.choice(growth_options) if growth_options else None

            print("  About to split", flush=True)
            split = plan_walk_forward_split(len(trial_graph.common_timestamps), model)
            if split is None:
                print("  Skipping: not enough bars for walk-forward split")
                continue
            _, train_end_bar, validation_end_bar = split

            loss_history = []
            total_loss, n_updates, _, loss_history = run_training(
                trial_graph,
                model,
                start_bar=0,
                end_bar=train_end_bar,
                print_every=100,
                loss_history=loss_history,
                use_profit_loss=(exchange == "binance"),
            )
            val_bpb, val_updates = run_walk_forward_validation(
                trial_graph,
                model,
                validation_start_bar=train_end_bar,
                end_bar=validation_end_bar,
                warmup_start_bar=0,
                use_profit_loss=(exchange == "binance"),
            )

            if n_updates > 0 and checkpoint_path:
                model.save(
                    checkpoint_path, checkpoint_type=f"autoresearch_phase_{phase}"
                )

            if val_bpb is not None:
                print(
                    f"  train_bpb={total_loss / n_updates:.6f} walk_forward_bpb={val_bpb:.6f} (best={best_bpb:.6f})"
                )
                if val_bpb < best_bpb:
                    best_bpb = val_bpb
                    print(f"  --> [NEW BEST] val_bpb: {best_bpb:.6f}")
                    if checkpoint_path:
                        model.save(
                            _checkpoint_variant(checkpoint_path, "best"),
                            checkpoint_type=f"autoresearch_best_phase_{phase}",
                        )
                with duckdb.connect(experiments_db_path) as conn:
                    conn.execute(
                        "INSERT INTO experiments (timestamp, val_bpb, params, bag_spec, growth_phase, model_cas) VALUES (now(), ?, ?, ?, ?, ?)",
                        [
                            val_bpb,
                            str(
                                {
                                    "lr": lr,
                                    "h_dim": hidden_size,
                                    "y_depth": y_depth,
                                    "x_pixels": x_pixels,
                                    "curvature": curvature,
                                    "prediction_depth": prediction_depth,
                                    "H_layers": H_layers,
                                    "L_layers": L_layers,
                                }
                            ),
                            str(
                                {
                                    "n_pairs": len(selected_pairs),
                                    "window_bars": window_bars,
                                    "exchange": exchange,
                                }
                            ),
                            growth_dim,
                            model.model_cas_signature(),
                        ],
                    )

            if growth_dim is not None and _is_converged(loss_history):
                old_h, old_H, old_L, old_Hc, old_Lc = (
                    hidden_size,
                    H_layers,
                    L_layers,
                    H_cycles,
                    L_cycles,
                )
                hidden_size, H_layers, L_layers, H_cycles, L_cycles, did_grow = (
                    _apply_growth_step(
                        model,
                        growth_dim,
                        hidden_size,
                        H_layers,
                        L_layers,
                        H_cycles,
                        L_cycles,
                    )
                )
                if did_grow:
                    print(
                        f"\n  *** CONVERGED -> GROWTH: {growth_dim}[{old_h}, {old_H}, {old_L}, {old_Hc}, {old_Lc}] ->[{hidden_size}, {H_layers}, {L_layers}, {H_cycles}, {L_cycles}]"
                    )
                    if checkpoint_path:
                        model.save(
                            checkpoint_path,
                            checkpoint_type=f"autoresearch_grown_phase_{phase}",
                        )
                else:
                    print(
                        f"\n  *** CONVERGED -> NO GROWTH: {growth_dim} already at max"
                    )
    except KeyboardInterrupt:
        print("\nInterrupted.")


# --- Finetune ---
def finetune(
    pretrained_path: str,
    bag_path: str,
    output_path: str = "model_weights_daytrade.pt",
    learning_rate: float = 0.0001,
    lookback_days: int = 60,
    exchange: str = "coinbase",
    skip_fetch: bool = True,
    device: str = "cpu",
):
    print("=" * 60 + "\nHRM Fine-Tuning for Daytrading\n" + "=" * 60)
    with open(bag_path, "r") as f:
        selected_pairs = json.load(f)
    print(
        f"\nUsing fixed bag: {len(selected_pairs)} pairs\nFine-tuning on last {lookback_days} days of data\nLearning rate: {learning_rate}\nDevice: {device}\nGrowth: DISABLED"
    )

    start_time = datetime.now() - timedelta(days=lookback_days + 30)
    trial_graph = CoinGraph(fee_rate=0.001)
    n_bars = trial_graph.load(
        exchange=exchange,
        skip_fetch=skip_fetch,
        start_time=start_time,
        end_time=datetime.now(),
        explicit_bag=[
            {"exchange": exchange, "product_id": pid} for pid in selected_pairs
        ],
    )
    print(
        f"Loaded {len(trial_graph.nodes)} nodes, {len(trial_graph.edges)} edges, {n_bars} bars"
    )
    if not trial_graph.edges:
        raise ValueError("No edges available in bag subgraph")

    print("  About to init model", flush=True)
    model = HierarchicalReasoningModel(
        n_edges=len(trial_graph.edges),
        learning_rate=learning_rate,
        y_depth=200,
        x_pixels=20,
        curvature=2.0,
        h_dim=4,
        z_dim=4,
        prediction_depth=1,
        H_layers=2,
        L_layers=2,
        device=device,
    )
    print("  About to register edges", flush=True)
    model.register_edges(_canonical_pair_edges(trial_graph))
    if Path(pretrained_path).exists():
        model.load(pretrained_path)
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    model._lr = learning_rate
    if model._optimizer:
        for param_group in model._optimizer.param_groups:
            param_group["lr"] = learning_rate

    print("  About to split", flush=True)
    split = plan_walk_forward_split(len(trial_graph.common_timestamps), model)
    if split is None:
        raise ValueError("Not enough bars for walk-forward fine-tuning validation")
    start_bar, train_end_bar, validation_end_bar = split
    print(
        f"\nFine-tuning on bars {start_bar}:{train_end_bar} with walk-forward holdout {train_end_bar}:{validation_end_bar}"
    )

    total_loss, n_updates, _, _ = run_training(
        trial_graph, model, start_bar=start_bar, end_bar=train_end_bar, print_every=100
    )
    walk_forward_loss, walk_forward_updates = run_walk_forward_validation(
        trial_graph,
        model,
        validation_start_bar=train_end_bar,
        end_bar=validation_end_bar,
        warmup_start_bar=start_bar,
    )

    print(f"\nFine-tuning complete: {_training_status(total_loss, n_updates)}")
    if walk_forward_loss is not None:
        print(
            f"Walk-forward holdout: avg_loss={walk_forward_loss:.6f}, n_updates={walk_forward_updates}"
        )
    if n_updates > 0 and walk_forward_loss is not None:
        model.save(output_path, checkpoint_type="finetuned_daytrade")
        print(f"Saved fine-tuned model to: {output_path}")


# --- Worker ---
class TrainingWorker:
    def __init__(
        self,
        mode: str,
        worker_id: str = "worker",
        checkpoint_dir: str = "./checkpoints",
        save_every: int = 1000,
        poll_interval: int = 60,
        lookback_days: int = 60,
        device: str = "cpu",
    ):
        (
            self.mode,
            self.worker_id,
            self.checkpoint_dir,
            self.save_every,
            self.poll_interval,
            self.lookback_days,
            self.device,
        ) = (
            mode,
            worker_id,
            Path(checkpoint_dir),
            save_every,
            poll_interval,
            lookback_days,
            device,
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.worker_id)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"./logs/{self.worker_id}.{self.mode}.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.model, self.phase, self.stopped, self.fixed_bag = None, 0, False, None
        self.hidden_size, self.H_layers, self.L_layers = SQUARE_CUBE_SIZES[0], 1, 1
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, "stopped", True))
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, "stopped", True))

    def _get_checkpoint_path(self, checkpoint_type: str = None) -> Path:
        return self.checkpoint_dir / (
            "model_weights_pretrained.pt"
            if self.mode == "pretrain" or checkpoint_type == "pretrained"
            else f"model_weights_{self.worker_id}.pt"
        )

    def _run_pretrain_cycle(self):
        self.phase += 1
        rng = random.Random()
        if _use_pool_for_db(str(Config.DB_PATH)):
            rows = _pool().execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE exchange = 'binance'"
            )
        else:
            with duckdb.connect(str(Config.DB_PATH), read_only=True) as conn:
                rows = conn.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE exchange = 'binance'"
                ).fetchall()
        db_min_ts = (
            pd.Timestamp(
                rows[0][0]
                if rows and rows[0][0]
                else _utc_now_naive() - timedelta(days=365)
            )
            .to_pydatetime()
            .replace(tzinfo=None)
        )
        db_max_ts = (
            pd.Timestamp(rows[0][1] if rows and rows[0][1] else _utc_now_naive())
            .to_pydatetime()
            .replace(tzinfo=None)
        )
        total_seconds = (db_max_ts - db_min_ts).total_seconds()

        all_subs = _list_all_exchange_subscriptions(
            str(Config.DB_PATH), exchange="binance"
        )
        if not all_subs:
            return
        filtered = _compute_volatility_filter(
            str(Config.DB_PATH),
            all_subs,
            lookback_days=1095,
            granularity="300",
            min_velocity=0.001,
        )
        if not filtered:
            return
        selected_pairs = _stochastic_bag_sample(
            [s["product_id"] for s in filtered], self.hidden_size, rng
        )

        window_bars = min(10000, int(total_seconds / 300))
        start_offset = rng.uniform(0, max(0, total_seconds - window_bars * 300))
        start_time, end_time = (
            db_min_ts + timedelta(seconds=start_offset),
            db_min_ts + timedelta(seconds=start_offset + window_bars * 300),
        )

        trial_graph = CoinGraph(fee_rate=0.001)
        trial_graph.load(
            db_path=str(Config.DB_PATH),
            exchange="binance",
            skip_fetch=True,
            start_time=start_time,
            end_time=end_time,
            explicit_bag=[
                {"exchange": "binance", "product_id": pid} for pid in selected_pairs
            ],
        )
        if not trial_graph.edges:
            return

        cp_path = self._get_checkpoint_path()
        self.model = HierarchicalReasoningModel(
            n_edges=len(trial_graph.edges),
            h_dim=self.hidden_size,
            H_layers=self.H_layers,
            L_layers=self.L_layers,
            device=self.device,
        )
        self.print("  About to register edges", flush=True)
        self.model.register_edges(_canonical_pair_edges(trial_graph))
        if cp_path.exists():
            self.model.load(str(cp_path))
            self.hidden_size, self.H_layers, self.L_layers = (
                self.model.h_dim,
                self.model.H_layers,
                self.model.L_layers,
            )

        growth_options = _growable_growth_dims(
            (("h", self.hidden_size), ("H", self.H_layers), ("L", self.L_layers))
        )
        growth_dim = rng.choice(growth_options) if growth_options else None

        print("  About to split", flush=True)
        split = plan_walk_forward_split(len(trial_graph.common_timestamps), self.model)
        if not split:
            return
        _, train_end_bar, validation_end_bar = split

        total_loss, n_updates, _, loss_history = run_training(
            trial_graph,
            self.model,
            start_bar=0,
            end_bar=train_end_bar,
            print_every=1000,
        )
        val_loss, val_updates = run_walk_forward_validation(
            trial_graph,
            self.model,
            validation_start_bar=train_end_bar,
            end_bar=validation_end_bar,
        )

        if growth_dim is not None and _is_converged(loss_history):
            self.hidden_size, self.H_layers, self.L_layers, _, _, did_grow = (
                _apply_growth_step(
                    self.model,
                    growth_dim,
                    self.hidden_size,
                    self.H_layers,
                    self.L_layers,
                )
            )
            if did_grow:
                self.model.save(
                    str(cp_path), checkpoint_type=f"pretrain_growth_{self.phase}"
                )
        elif self.phase % self.save_every == 0 and n_updates > 0:
            self.model.save(str(cp_path), checkpoint_type=f"pretrain_{self.phase}")

    def _run_finetune_cycle(self):
        self.phase += 1
        if not self.fixed_bag:
            with open(Config.BAG_PATH, "r") as f:
                self.fixed_bag = json.load(f)

        start_time, end_time = (
            datetime.now() - timedelta(days=self.lookback_days + 30),
            datetime.now(),
        )
        trial_graph = CoinGraph(fee_rate=0.001)
        trial_graph.load(
            db_path=str(Config.DB_PATH),
            exchange="coinbase",
            skip_fetch=True,
            start_time=start_time,
            end_time=end_time,
            explicit_bag=[
                {"exchange": "coinbase", "product_id": pid} for pid in self.fixed_bag
            ],
        )
        if not trial_graph.edges:
            return

        cp_path = self._get_checkpoint_path(checkpoint_type="pretrained")
        if not self.model:
            if not cp_path.exists():
                return
            self.model = HierarchicalReasoningModel(
                n_edges=len(trial_graph.edges),
                h_dim=4,
                H_layers=2,
                L_layers=2,
                device=self.device,
            )
            self.model.load(str(cp_path))
            self.model._lr = 0.0001

        self.print("  About to register edges", flush=True)
        self.model.register_edges(_canonical_pair_edges(trial_graph))
        if self.model._optimizer:
            for param_group in self.model._optimizer.param_groups:
                param_group["lr"] = 0.0001

        print("  About to split", flush=True)
        split = plan_walk_forward_split(len(trial_graph.common_timestamps), self.model)
        if not split:
            return
        _, train_end_bar, validation_end_bar = split

        total_loss, n_updates, _, _ = run_training(
            trial_graph, self.model, start_bar=0, end_bar=train_end_bar, print_every=100
        )
        val_loss, val_updates = run_walk_forward_validation(
            trial_graph,
            self.model,
            validation_start_bar=train_end_bar,
            end_bar=validation_end_bar,
        )

        if self.phase % self.save_every == 0 and n_updates > 0:
            self.model.save(
                str(self._get_checkpoint_path()),
                checkpoint_type=f"finetune_{self.phase}",
            )

    def run(self):
        while not self.stopped:
            try:
                if self.mode == "pretrain":
                    self._run_pretrain_cycle()
                elif self.mode == "finetune":
                    self._run_finetune_cycle()
            except Exception as e:
                self.logger.exception(f"Error: {e}")
            if not self.stopped:
                time.sleep(self.poll_interval)


# --- Orchestrator & Dashboard ---
def run_orchestrator():
    def prune():
        for pattern in ["model_weights_pretrained_*.pt", "model_weights_daytrade_*.pt"]:
            cps = sorted(
                Path("./checkpoints").glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in cps[10:]:
                try:
                    old.unlink()
                except Exception:
                    pass

    schedule.every().day.at("04:00").do(prune)
    while True:
        schedule.run_pending()
        time.sleep(60)


app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    return jsonify(
        {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "db_exists": Path(Config.DB_PATH).exists(),
        }
    )


@app.route("/api/metrics")
def api_metrics():
    return jsonify(
        {"current_phase": 0, "last_loss": None, "phases": [], "checkpoints": []}
    )


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="HRM Showdown Unified CLI")

    # Commands (as flags)
    parser.add_argument("--autoresearch", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--orchestrator", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--health", action="store_true")

    # Arguments
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-path", default="model_weights.pt")
    parser.add_argument("--training-db-path", default="training.duckdb")
    parser.add_argument("--dim", type=int, default=None)

    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--bag", default=None)
    parser.add_argument("--output", default="model_weights_daytrade.pt")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lookback-days", type=int, default=60)

    parser.add_argument("--mode", choices=["pretrain", "finetune"], default=None)
    parser.add_argument("--worker-id", default="worker")

    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    ensure_pool_running(str(Config.DB_PATH))

    if args.autoresearch:
        run_autoresearch(
            db_path=str(Config.DB_PATH),
            exchange=args.exchange,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            training_db_path=args.training_db_path,
            fixed_dim=args.dim,
        )
    elif args.finetune:
        if not args.pretrained or not args.bag:
            parser.error("--pretrained and --bag are required for --finetune")
        finetune(
            pretrained_path=args.pretrained,
            bag_path=args.bag,
            output_path=args.output,
            learning_rate=args.lr,
            lookback_days=args.lookback_days,
            exchange=args.exchange,
            skip_fetch=True,
            device=args.device,
        )
    elif args.worker:
        if not args.mode:
            parser.error("--mode is required for --worker")
        TrainingWorker(
            mode=args.mode, worker_id=args.worker_id, device=args.device
        ).run()
    elif args.orchestrator:
        run_orchestrator()
    elif args.dashboard:
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    elif args.health:
        app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
