from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EdgeKey = Tuple[str, ...]


def _parse_edge(edge: EdgeKey) -> Tuple[Optional[str], str, str]:
    if len(edge) == 2:
        base, quote = edge
        return None, str(base), str(quote)
    if len(edge) == 3:
        exchange, base, quote = edge
        return str(exchange), str(base), str(quote)
    raise ValueError(f"Unsupported edge shape: {edge!r}")


def _ordered_node_names(
    nodes: Iterable[str],
    node_to_idx: Optional[Mapping[str, int]] = None,
    value_asset: str = "USD",
) -> list[str]:
    if node_to_idx:
        return [name for name, _ in sorted(node_to_idx.items(), key=lambda item: item[1])]
    node_names = sorted({str(node) for node in nodes})
    if value_asset in node_names:
        node_names.remove(value_asset)
        node_names.insert(0, value_asset)
    return node_names


def _build_node_to_idx(
    nodes: Iterable[str],
    node_to_idx: Optional[Mapping[str, int]] = None,
    value_asset: str = "USD",
) -> Dict[str, int]:
    if node_to_idx:
        return {str(name): int(idx) for name, idx in node_to_idx.items()}
    node_names = _ordered_node_names(nodes, value_asset=value_asset)
    return {name: idx for idx, name in enumerate(node_names)}


def _normalize_balance_map(
    values: Optional[Any],
    *,
    node_names: Sequence[str],
    node_to_idx: Mapping[str, int],
    value_column: str,
) -> Dict[str, float]:
    normalized = {name: 0.0 for name in node_names}
    if values is None:
        return normalized

    if isinstance(values, pd.DataFrame):
        if {"coin_idx", value_column}.issubset(values.columns):
            for row in values[["coin_idx", value_column]].itertuples(index=False):
                idx = int(row.coin_idx)
                if 0 <= idx < len(node_names):
                    normalized[node_names[idx]] = float(getattr(row, value_column))
            return normalized
        asset_column = "asset" if "asset" in values.columns else "symbol" if "symbol" in values.columns else None
        if asset_column and value_column in values.columns:
            for row in values[[asset_column, value_column]].itertuples(index=False):
                asset = str(getattr(row, asset_column))
                if asset in normalized:
                    normalized[asset] = float(getattr(row, value_column))
            return normalized
        raise ValueError(
            f"Unsupported balance DataFrame shape; expected coin_idx/{value_column} or asset/{value_column}"
        )

    if isinstance(values, Mapping):
        for key, value in values.items():
            if isinstance(key, int):
                if 0 <= key < len(node_names):
                    normalized[node_names[key]] = float(value)
                continue
            asset = str(key)
            if asset in normalized:
                normalized[asset] = float(value)
        return normalized

    raise TypeError(f"Unsupported balance type for {value_column}: {type(values).__name__}")


def _directional_prices(graph, edge: EdgeKey, row) -> Dict[str, float]:
    if hasattr(graph, "edge_price_components"):
        return graph.edge_price_components(edge, row)

    close = float(row.get("close", 0.0) or 0.0)
    open_price = float(row.get("open", close) or close or 0.0)
    high = float(row.get("high", close) or close or 0.0)
    low = float(row.get("low", close) or close or 0.0)
    is_inverted = bool(getattr(graph, "edge_is_inverted", {}).get(edge, False))
    if not is_inverted:
        return {"open": open_price, "high": high, "low": low, "close": close}

    inv_open = 1.0 / open_price if open_price > 0 else 0.0
    inv_close = 1.0 / close if close > 0 else 0.0
    inv_high = 1.0 / low if low > 0 else 0.0
    inv_low = 1.0 / high if high > 0 else 0.0
    return {"open": inv_open, "high": inv_high, "low": inv_low, "close": inv_close}


def _carry_features(carry: Any) -> Dict[str, float]:
    if carry is None:
        return {}
    features: Dict[str, float] = {}
    z_h = getattr(carry, "z_H", None)
    z_l = getattr(carry, "z_L", None)
    if z_h is not None:
        values = np.asarray(z_h.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        for idx, value in enumerate(values):
            features[f"carry_h_{idx}"] = float(value)
    if z_l is not None:
        values = np.asarray(z_l.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        for idx, value in enumerate(values):
            features[f"carry_l_{idx}"] = float(value)
    return features


def _coerce_route_discount(route_discount: Optional[Any], edge: EdgeKey) -> float:
    if route_discount is None:
        return 0.0
    if isinstance(route_discount, Mapping):
        return float(route_discount.get(edge, 0.0))
    return float(route_discount)


def _route_to_value_asset(
    pair_state_df: pd.DataFrame,
    *,
    node_names: Sequence[str],
    node_to_idx: Mapping[str, int],
    value_asset: str,
    discounted: bool,
) -> Dict[int, Dict[str, Any]]:
    value_idx = node_to_idx.get(value_asset)
    if value_idx is None:
        return {}

    price_column = "discounted_price" if discounted else "price_now"
    neg_inf = float("-inf")
    best_log_value = {idx: neg_inf for idx in range(len(node_names))}
    best_log_value[value_idx] = 0.0
    best_depth = {value_idx: 0}
    best_discount = {value_idx: 0.0}
    best_path = {value_idx: value_asset}

    for _ in range(max(0, len(node_names) - 1)):
        changed = False
        for row in pair_state_df.itertuples(index=False):
            src_idx = int(row.base_idx)
            dst_idx = int(row.quote_idx)
            rate = float(getattr(row, price_column))
            if rate <= 0.0 or best_log_value[dst_idx] == neg_inf:
                continue

            candidate_log_value = math.log(rate) + best_log_value[dst_idx]
            candidate_depth = best_depth[dst_idx] + 1
            candidate_discount = best_discount[dst_idx] + float(getattr(row, "edge_discount_total"))

            should_replace = candidate_log_value > best_log_value[src_idx] + 1e-12
            if not should_replace and math.isclose(candidate_log_value, best_log_value[src_idx], abs_tol=1e-12):
                should_replace = candidate_depth < best_depth.get(src_idx, math.inf)

            if should_replace:
                best_log_value[src_idx] = candidate_log_value
                best_depth[src_idx] = candidate_depth
                best_discount[src_idx] = candidate_discount
                best_path[src_idx] = f"{node_names[src_idx]}->{best_path[dst_idx]}"
                changed = True
        if not changed:
            break

    routes: Dict[int, Dict[str, Any]] = {}
    for idx, log_value in best_log_value.items():
        if log_value == neg_inf:
            continue
        routes[idx] = {
            "cost": -log_value,
            "depth": best_depth.get(idx, -1),
            "discount": best_discount.get(idx, 0.0),
            "path": best_path.get(idx),
        }
    return routes


def _node_route_frame(
    pair_state_df: pd.DataFrame,
    *,
    node_names: Sequence[str],
    node_to_idx: Mapping[str, int],
    value_asset: str,
    ts,
    free_qty: Optional[Any],
    reserved_qty: Optional[Any],
) -> pd.DataFrame:
    raw_routes = _route_to_value_asset(
        pair_state_df,
        node_names=node_names,
        node_to_idx=node_to_idx,
        value_asset=value_asset,
        discounted=False,
    )
    discounted_routes = _route_to_value_asset(
        pair_state_df,
        node_names=node_names,
        node_to_idx=node_to_idx,
        value_asset=value_asset,
        discounted=True,
    )

    free_map = _normalize_balance_map(
        free_qty,
        node_names=node_names,
        node_to_idx=node_to_idx,
        value_column="free_qty",
    )
    reserved_map = _normalize_balance_map(
        reserved_qty,
        node_names=node_names,
        node_to_idx=node_to_idx,
        value_column="reserved_qty",
    )

    rows = []
    for coin_idx, asset in enumerate(node_names):
        raw_route = raw_routes.get(coin_idx)
        discounted_route = discounted_routes.get(coin_idx)
        root_price = math.exp(-raw_route["cost"]) if raw_route else 0.0
        discounted_root_price = math.exp(-discounted_route["cost"]) if discounted_route else 0.0
        free_value = free_map[asset]
        reserved_value = reserved_map[asset]
        rows.append(
            {
                "ts": ts,
                "coin_idx": coin_idx,
                "asset": asset,
                "is_value_asset": asset == value_asset,
                "root_price": root_price,
                "discounted_root_price": discounted_root_price,
                "route_depth": int(discounted_route["depth"]) if discounted_route else -1,
                "route_discount": float(discounted_route["discount"]) if discounted_route else 0.0,
                "route_path": discounted_route["path"] if discounted_route else None,
                "free_qty": free_value,
                "reserved_qty": reserved_value,
                "inventory_value": free_value * discounted_root_price,
            }
        )
    return pd.DataFrame(rows).sort_values("coin_idx").reset_index(drop=True)


def _enrich_pair_state(pair_state_df: pd.DataFrame, node_state_df: pd.DataFrame) -> pd.DataFrame:
    anchors = node_state_df[
        [
            "coin_idx",
            "asset",
            "root_price",
            "discounted_root_price",
            "route_depth",
            "route_discount",
            "inventory_value",
            "free_qty",
            "reserved_qty",
        ]
    ]
    base = anchors.rename(
        columns={
            "coin_idx": "base_idx",
            "asset": "base_asset_anchor",
            "root_price": "base_root_price",
            "discounted_root_price": "base_discounted_root_price",
            "route_depth": "base_route_depth",
            "route_discount": "base_route_discount",
            "inventory_value": "base_inventory_value",
            "free_qty": "base_free_qty",
            "reserved_qty": "base_reserved_qty",
        }
    )
    quote = anchors.rename(
        columns={
            "coin_idx": "quote_idx",
            "asset": "quote_asset_anchor",
            "root_price": "quote_root_price",
            "discounted_root_price": "quote_discounted_root_price",
            "route_depth": "quote_route_depth",
            "route_discount": "quote_route_discount",
            "inventory_value": "quote_inventory_value",
            "free_qty": "quote_free_qty",
            "reserved_qty": "quote_reserved_qty",
        }
    )
    enriched = pair_state_df.merge(base, on="base_idx", how="left").merge(quote, on="quote_idx", how="left")
    enriched["price_prime"] = np.where(
        enriched["quote_root_price"] > 0,
        enriched["base_root_price"] / enriched["quote_root_price"],
        0.0,
    )
    enriched["price_prime_discounted"] = np.where(
        enriched["quote_discounted_root_price"] > 0,
        enriched["base_discounted_root_price"] / enriched["quote_discounted_root_price"],
        0.0,
    )
    enriched["price_vs_prime"] = np.where(
        enriched["price_prime"] > 0,
        enriched["price_now"] / enriched["price_prime"] - 1.0,
        0.0,
    )
    return enriched


def build_pair_state_df(
    graph,
    *,
    bar_idx: int,
    edge_names: Optional[Sequence[EdgeKey]] = None,
    node_to_idx: Optional[Mapping[str, int]] = None,
    value_asset: str = "USD",
    edge_fisheyes: Optional[Mapping[EdgeKey, Sequence[float]]] = None,
    edge_carries: Optional[Mapping[EdgeKey, Any]] = None,
    route_discount: Optional[Any] = None,
) -> pd.DataFrame:
    if bar_idx < 0 or bar_idx >= len(getattr(graph, "common_timestamps", [])):
        return pd.DataFrame()

    ts = graph.common_timestamps[bar_idx]
    edge_names = list(edge_names or getattr(graph, "edges", {}).keys())
    node_to_idx = _build_node_to_idx(getattr(graph, "nodes", []), node_to_idx=node_to_idx, value_asset=value_asset)
    rows = []
    fisheye_map = edge_fisheyes or {}
    carry_map = edge_carries or {}

    for edge in edge_names:
        df = graph.edges.get(edge)
        if df is None or ts not in df.index:
            continue
        row = df.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        exchange, base_asset, quote_asset = _parse_edge(edge)
        prices = _directional_prices(graph, edge, row)
        price_now = float(prices["close"])
        edge_discount = _coerce_route_discount(route_discount, edge)
        fee_rate = float(getattr(graph, "fee_rate", 0.0))
        edge_discount_total = fee_rate + edge_discount
        discounted_price = price_now * math.exp(-edge_discount_total) if price_now > 0 else 0.0

        edge_state = getattr(graph, "edge_state", {}).get(edge)
        payload: Dict[str, Any] = {
            "ts": ts,
            "exchange": exchange,
            "pair_key": f"{node_to_idx[base_asset]}-{node_to_idx[quote_asset]}",
            "pair_id": f"{base_asset}->{quote_asset}",
            "product_id": getattr(graph, "edge_product_id", {}).get(edge),
            "is_inverted": bool(getattr(graph, "edge_is_inverted", {}).get(edge, False)),
            "base_idx": node_to_idx[base_asset],
            "quote_idx": node_to_idx[quote_asset],
            "base_asset": base_asset,
            "quote_asset": quote_asset,
            "price_open": float(prices["open"]),
            "price_high": float(prices["high"]),
            "price_low": float(prices["low"]),
            "price_now": price_now,
            "price_return": math.log(price_now / prices["open"]) if prices["open"] > 0 and price_now > 0 else 0.0,
            "edge_velocity": float(getattr(edge_state, "velocity", 0.0) or 0.0),
            "edge_ptt": float(getattr(edge_state, "ptt", 0.0) or 0.0),
            "edge_stop": float(getattr(edge_state, "stop", 0.0) or 0.0),
            "edge_hit_ptt": bool(getattr(edge_state, "hit_ptt", False)),
            "edge_hit_stop": bool(getattr(edge_state, "hit_stop", False)),
            "fee_rate": fee_rate,
            "route_discount": edge_discount,
            "edge_discount_total": edge_discount_total,
            "discounted_price": discounted_price,
        }

        for idx, value in enumerate(fisheye_map.get(edge, ())):
            payload[f"fisheye_{idx}"] = float(value)
        payload.update(_carry_features(carry_map.get(edge)))
        rows.append(payload)

    if not rows:
        return pd.DataFrame()

    pair_state_df = pd.DataFrame(rows).sort_values(["base_idx", "quote_idx"]).reset_index(drop=True)
    node_names = _ordered_node_names(node_to_idx.keys(), node_to_idx=node_to_idx, value_asset=value_asset)
    node_state_df = _node_route_frame(
        pair_state_df,
        node_names=node_names,
        node_to_idx=node_to_idx,
        value_asset=value_asset,
        ts=ts,
        free_qty=None,
        reserved_qty=None,
    )
    return _enrich_pair_state(pair_state_df, node_state_df)


def build_node_state_df(
    pair_state_df: pd.DataFrame,
    *,
    node_to_idx: Mapping[str, int],
    value_asset: str = "USD",
    free_qty: Optional[Any] = None,
    reserved_qty: Optional[Any] = None,
) -> pd.DataFrame:
    if pair_state_df.empty:
        return pd.DataFrame()
    node_names = _ordered_node_names(node_to_idx.keys(), node_to_idx=node_to_idx, value_asset=value_asset)
    ts = pair_state_df["ts"].iloc[0]
    return _node_route_frame(
        pair_state_df,
        node_names=node_names,
        node_to_idx=node_to_idx,
        value_asset=value_asset,
        ts=ts,
        free_qty=free_qty,
        reserved_qty=reserved_qty,
    )


def build_model_df(node_state_df: pd.DataFrame, pair_state_df: pd.DataFrame) -> pd.DataFrame:
    if pair_state_df.empty:
        return pd.DataFrame()
    if node_state_df.empty:
        return pair_state_df.copy()

    merge_columns = [
        "coin_idx",
        "asset",
        "root_price",
        "discounted_root_price",
        "route_depth",
        "route_discount",
        "route_path",
        "free_qty",
        "reserved_qty",
        "inventory_value",
    ]
    base = node_state_df[merge_columns].rename(
        columns={
            "coin_idx": "base_idx",
            "asset": "base_asset_node",
            "root_price": "base_node_root_price",
            "discounted_root_price": "base_node_discounted_root_price",
            "route_depth": "base_node_route_depth",
            "route_discount": "base_node_route_discount",
            "route_path": "base_node_route_path",
            "free_qty": "base_node_free_qty",
            "reserved_qty": "base_node_reserved_qty",
            "inventory_value": "base_node_inventory_value",
        }
    )
    quote = node_state_df[merge_columns].rename(
        columns={
            "coin_idx": "quote_idx",
            "asset": "quote_asset_node",
            "root_price": "quote_node_root_price",
            "discounted_root_price": "quote_node_discounted_root_price",
            "route_depth": "quote_node_route_depth",
            "route_discount": "quote_node_route_discount",
            "route_path": "quote_node_route_path",
            "free_qty": "quote_node_free_qty",
            "reserved_qty": "quote_node_reserved_qty",
            "inventory_value": "quote_node_inventory_value",
        }
    )
    return (
        pair_state_df
        .merge(base, on="base_idx", how="left")
        .merge(quote, on="quote_idx", how="left")
        .sort_values(["base_idx", "quote_idx"])
        .reset_index(drop=True)
    )


def build_bag_model_frames(
    graph,
    *,
    bar_idx: int,
    edge_names: Sequence[EdgeKey],
    node_to_idx: Mapping[str, int],
    value_asset: str = "USD",
    edge_fisheyes: Optional[Mapping[EdgeKey, Sequence[float]]] = None,
    edge_carries: Optional[Mapping[EdgeKey, Any]] = None,
    free_qty: Optional[Any] = None,
    reserved_qty: Optional[Any] = None,
    route_discount: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_state_df = build_pair_state_df(
        graph,
        bar_idx=bar_idx,
        edge_names=edge_names,
        node_to_idx=node_to_idx,
        value_asset=value_asset,
        edge_fisheyes=edge_fisheyes,
        edge_carries=edge_carries,
        route_discount=route_discount,
    )
    if pair_state_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    node_state_df = build_node_state_df(
        pair_state_df,
        node_to_idx=node_to_idx,
        value_asset=value_asset,
        free_qty=free_qty,
        reserved_qty=reserved_qty,
    )
    pair_state_df = _enrich_pair_state(
        pair_state_df.drop(
            columns=[
                column
                for column in (
                    "base_asset_anchor",
                    "base_root_price",
                    "base_discounted_root_price",
                    "base_route_depth",
                    "base_route_discount",
                    "base_inventory_value",
                    "base_free_qty",
                    "base_reserved_qty",
                    "quote_asset_anchor",
                    "quote_root_price",
                    "quote_discounted_root_price",
                    "quote_route_depth",
                    "quote_route_discount",
                    "quote_inventory_value",
                    "quote_free_qty",
                    "quote_reserved_qty",
                    "price_prime",
                    "price_prime_discounted",
                    "price_vs_prime",
                )
                if column in pair_state_df.columns
            ]
        ),
        node_state_df,
    )
    model_df = build_model_df(node_state_df, pair_state_df)
    return node_state_df, pair_state_df, model_df


def bind_spend_budgets(
    pred_df: pd.DataFrame,
    node_state_df: pd.DataFrame,
    *,
    hi_col: str = "hi",
    lo_col: str = "lo",
    fraction_col: str = "fraction",
) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()

    required_columns = {"base_idx", "quote_idx", hi_col, lo_col, fraction_col}
    missing = required_columns - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")

    inventory_df = node_state_df[["coin_idx", "asset", "free_qty"]].rename(
        columns={
            "coin_idx": "spend_idx",
            "asset": "spend_asset",
            "free_qty": "available_spend_qty",
        }
    )

    exec_df = pred_df.copy()
    exec_df["fraction"] = exec_df[fraction_col].clip(lower=0.0, upper=1.0)
    exec_df["polarity"] = np.where(exec_df[hi_col] >= exec_df[lo_col], "hi", "lo")
    exec_df["spend_idx"] = np.where(exec_df["polarity"].eq("hi"), exec_df["base_idx"], exec_df["quote_idx"])
    exec_df = exec_df.merge(inventory_df, on="spend_idx", how="left")
    exec_df["available_spend_qty"] = exec_df["available_spend_qty"].fillna(0.0)
    exec_df["requested_spend_qty"] = exec_df["fraction"] * exec_df["available_spend_qty"]

    total_claim = exec_df.groupby("spend_idx")["requested_spend_qty"].transform("sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        clip_scale = np.where(
            total_claim > exec_df["available_spend_qty"],
            np.where(total_claim > 0, exec_df["available_spend_qty"] / total_claim, 1.0),
            1.0,
        )
    exec_df["clip_scale"] = clip_scale
    exec_df["final_spend_qty"] = exec_df["requested_spend_qty"] * exec_df["clip_scale"]

    if "price_now" in exec_df.columns:
        exec_df["parent_base_qty"] = np.where(
            exec_df["polarity"].eq("hi"),
            exec_df["final_spend_qty"],
            np.where(exec_df["price_now"] > 0, exec_df["final_spend_qty"] / exec_df["price_now"], 0.0),
        )
        exec_df["parent_quote_qty"] = np.where(
            exec_df["polarity"].eq("hi"),
            exec_df["final_spend_qty"] * exec_df["price_now"],
            exec_df["final_spend_qty"],
        )
    return exec_df
