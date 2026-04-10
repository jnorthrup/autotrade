from __future__ import annotations

import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from cache import _finite_float

EdgeKey = Tuple[str, ...]


def _parse_edge(edge: EdgeKey) -> Tuple[Optional[str], str, str]:
    if len(edge) == 2:
        return None, str(edge[0]), str(edge[1])
    if len(edge) == 3:
        return str(edge[0]), str(edge[1]), str(edge[2])
    raise ValueError(f"Unsupported edge shape: {edge!r}")


def _canonical_graph_edges(graph: Any) -> List[EdgeKey]:
    canonical: List[EdgeKey] = []
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


class CircularHistory:
    def __init__(self, maxlen: int = 512):
        self.maxlen = max(1, int(maxlen))
        self._items: Deque[Any] = deque(maxlen=self.maxlen)

    def append(self, item: Any) -> None:
        self._items.append(item)

    def extend(self, items: Iterable[Any]) -> None:
        self._items.extend(items)

    def to_list(self) -> List[Any]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


@dataclass
class OrderShim:
    edge: EdgeKey
    price: float
    is_buy: bool
    amt: float
    created_bar: int
    maturity_bar: int
    spend_asset: str
    confidence: float
    fraction: float

    def __str__(self) -> str:
        _, base_asset, quote_asset = _parse_edge(self.edge)
        symbol = f"{base_asset}-{quote_asset}"
        side = "BUY" if self.is_buy else "SELL"
        return f"Order({symbol} @{self.price:.8f}, {side} amt:{self.amt:.8f}, mat:{self.maturity_bar})"


@dataclass
class AgentAssetBalance:
    asset: str
    free: float = 0.0
    order_list: List[OrderShim] = field(default_factory=list)

    @property
    def sim_locked(self) -> float:
        return sum(max(0.0, float(order.amt)) for order in self.order_list)

    @property
    def total(self) -> float:
        return max(0.0, float(self.free)) + self.sim_locked

    def __str__(self) -> str:
        return f"bal({self.asset}, free:{self.free:.8f}, lock:{self.sim_locked:.8f})"


class SimWallet:
    def __init__(self, assets: Iterable[str], *, value_asset: str, initial_balances: Optional[Mapping[str, float]] = None, history_size: int = 1024):
        self.value_asset = str(value_asset)
        self.holdings: Dict[str, AgentAssetBalance] = {}
        self.audit = CircularHistory(history_size)
        self.ledger = CircularHistory(history_size)
        self.snapshots = CircularHistory(history_size)
        self._worth_cache: Optional[Tuple[int, float]] = None
        self._route_cache: Optional[Tuple[int, Dict[str, Dict[str, Any]]]] = None
        for asset in sorted({str(a) for a in assets} | {self.value_asset}):
            self.add_asset(asset, (initial_balances or {}).get(asset, 0.0))
        self._record("wallet_init", bar_idx=-1, initial_balances=self.free_qty_map())

    def add_asset(self, asset: str, free: float = 0.0) -> None:
        self.holdings[str(asset)] = AgentAssetBalance(asset=str(asset), free=max(0.0, float(free)))

    def _record(self, event_type: str, **payload: Any) -> None:
        entry = {"event": event_type, **payload}
        self.ledger.append(entry)
        self.audit.append(f"{event_type}: {payload}")

    def balance(self, asset: str) -> AgentAssetBalance:
        asset = str(asset)
        if asset not in self.holdings:
            self.add_asset(asset, 0.0)
        return self.holdings[asset]

    def free_qty_map(self) -> Dict[str, float]:
        return {asset: bal.free for asset, bal in self.holdings.items()}

    def reserved_qty_map(self) -> Dict[str, float]:
        return {asset: bal.sim_locked for asset, bal in self.holdings.items()}

    def total_qty_map(self) -> Dict[str, float]:
        return {asset: bal.total for asset, bal in self.holdings.items()}

    @property
    def open_order_count(self) -> int:
        return sum(len(balance.order_list) for balance in self.holdings.values())

    def reserve(self, order: OrderShim) -> bool:
        balance = self.balance(order.spend_asset)
        spend_amt = max(0.0, min(balance.free, float(order.amt)))
        if spend_amt <= 0.0:
            return False
        order.amt = spend_amt
        balance.free -= spend_amt
        balance.order_list.append(order)
        self._worth_cache = None
        self._route_cache = None
        self._record(
            "order_reserved",
            bar_idx=order.created_bar,
            asset=order.spend_asset,
            amount=spend_amt,
            side="BUY" if order.is_buy else "SELL",
            edge=order.edge,
            maturity_bar=order.maturity_bar,
            free_after=balance.free,
            reserved_after=balance.sim_locked,
        )
        return True

    def cancel(self, order: OrderShim, *, bar_idx: int, reason: str = "cancelled") -> None:
        balance = self.balance(order.spend_asset)
        for idx, current in enumerate(balance.order_list):
            if current is order:
                balance.order_list.pop(idx)
                balance.free += max(0.0, float(order.amt))
                self._worth_cache = None
                self._route_cache = None
                self._record(
                    "order_cancelled",
                    bar_idx=bar_idx,
                    reason=reason,
                    asset=order.spend_asset,
                    amount=order.amt,
                    side="BUY" if order.is_buy else "SELL",
                    edge=order.edge,
                    free_after=balance.free,
                    reserved_after=balance.sim_locked,
                )
                return

    def repost_open_orders(self, *, bar_idx: int, reason: str = "repost") -> int:
        """Cancel all currently open (reserved) orders to allow re-posting new
        orders each bar (non-GTC workflow). Returns the number of orders
        cancelled.
        """
        cancelled = 0
        # iterate over a copy of holdings to avoid mutation during iteration
        for asset in list(self.holdings.keys()):
            balance = self.balance(asset)
            # copy list since cancel mutates order_list
            for order in list(balance.order_list):
                self.cancel(order, bar_idx=bar_idx, reason=reason)
                cancelled += 1
        if cancelled:
            # clear caches and snapshot for visibility
            self._worth_cache = None
            self._route_cache = None
            # No graph available here; caller may snapshot after placing new orders
        return cancelled

    def _edge_row(self, graph: Any, edge: EdgeKey, bar_idx: int) -> Optional[Any]:
        if bar_idx < 0 or bar_idx >= len(getattr(graph, "common_timestamps", [])):
            return None
        ts = graph.common_timestamps[bar_idx]
        df = getattr(graph, "edges", {}).get(edge)
        if df is None or ts not in df.index:
            return None
        row = df.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row

    def edge_price(self, graph: Any, edge: EdgeKey, bar_idx: int) -> float:
        row = self._edge_row(graph, edge, bar_idx)
        if row is None:
            return 0.0
        if hasattr(graph, "edge_price_components"):
            return _finite_float(graph.edge_price_components(edge, row).get("close", 0.0), 0.0)
        return _finite_float(row.get("close", 0.0), 0.0)

    def reserve_orders(self, orders: Iterable[OrderShim], *, graph: Any, bar_idx: int, label: str = "post_place") -> int:
        created = 0
        for order in orders:
            if self.reserve(order):
                created += 1
        if created:
            self.snapshot(graph, bar_idx=bar_idx, label=label)
        return created

    def settle_due_orders(self, graph: Any, *, bar_idx: int, fee_rate: float) -> Dict[str, Any]:
        worth_before = self.worth(graph, bar_idx)
        settled = 0
        for asset in list(self.holdings.keys()):
            balance = self.balance(asset)
            remaining: List[OrderShim] = []
            for order in balance.order_list:
                # Attempt intrabar execution: if the bar's high/low crosses the
                # order price, execute immediately using a realistic intrabar
                # exit price. Otherwise, only settle when maturity_bar <= bar_idx.
                row = self._edge_row(graph, order.edge, bar_idx)
                open_p = high_p = low_p = close_p = 0.0
                if row is not None and hasattr(graph, "edge_price_components"):
                    comps = graph.edge_price_components(order.edge, row)
                    open_p = _finite_float(comps.get("open", 0.0), 0.0)
                    high_p = _finite_float(comps.get("high", 0.0), 0.0)
                    low_p = _finite_float(comps.get("low", 0.0), 0.0)
                    close_p = _finite_float(comps.get("close", 0.0), 0.0)
                else:
                    # fallback to close-only prices
                    close_p = self.edge_price(graph, order.edge, bar_idx)
                    open_p = high_p = low_p = close_p

                executed = False
                exit_price = close_p

                # If intrabar price range touches the order price, treat as executed
                if order.maturity_bar > bar_idx:
                    # not yet matured; check intrabar triggers
                    if order.is_buy:
                        # buy executes if low <= order.price
                        if low_p > 0.0 and low_p <= float(order.price):
                            executed = True
                            # use a conservative intrabar fill price similar to other codepaths
                            exit_price = min(high_p, max(float(order.price), open_p))
                    else:
                        # sell executes if high >= order.price
                        if high_p > 0.0 and high_p >= float(order.price):
                            executed = True
                            exit_price = max(low_p, min(float(order.price), open_p))
                else:
                    # matured: settle at close (or best available component)
                    executed = True
                    exit_price = close_p

                if not executed:
                    remaining.append(order)
                    continue

                _, base_asset, quote_asset = _parse_edge(order.edge)

                if order.is_buy:
                    # BUY: spent quote (USDT), receive base (ETH/BTC/...)
                    # credit base asset after fee
                    entry_price = max(1e-30, float(order.price))
                    fee_mult = math.exp(-max(0.0, float(fee_rate)))
                    base_qty = (float(order.amt) / entry_price) * fee_mult
                    base_bal = self.balance(base_asset)
                    base_bal.free += base_qty
                    settled += 1
                    self._record(
                        "order_settled",
                        bar_idx=bar_idx,
                        asset=order.spend_asset,
                        side="BUY",
                        edge=order.edge,
                        spend_amt=order.amt,
                        base_asset=base_asset,
                        base_qty=base_qty,
                        entry_price=order.price,
                        exit_price=exit_price,
                        fee_mult=fee_mult,
                        free_quote_after=balance.free,
                        free_base_after=base_bal.free,
                    )
                else:
                    # SELL: spent base (ETH/BTC/...), receive quote (USDT)
                    entry_price = max(1e-30, float(order.price))
                    fee_mult = math.exp(-max(0.0, float(fee_rate)))
                    quote_qty = (float(order.amt) * exit_price / entry_price) * fee_mult
                    quote_bal = self.balance(quote_asset)
                    quote_bal.free += quote_qty
                    settled += 1
                    self._record(
                        "order_settled",
                        bar_idx=bar_idx,
                        asset=order.spend_asset,
                        side="SELL",
                        edge=order.edge,
                        spend_amt=order.amt,
                        quote_asset=quote_asset,
                        quote_qty=quote_qty,
                        entry_price=order.price,
                        exit_price=exit_price,
                        fee_mult=fee_mult,
                        free_base_after=balance.free,
                        free_quote_after=quote_bal.free,
                    )
            balance.order_list = remaining
        self._worth_cache = None
        self._route_cache = None
        worth_after = self.worth(graph, bar_idx)
        bar_pnl = worth_after - worth_before
        self.snapshot(graph, bar_idx=bar_idx, label="post_settle")
        return {"bar_pnl": bar_pnl, "worth_before": worth_before, "worth_after": worth_after, "settled_orders": settled}

    def _route_quotes(self, graph: Any, bar_idx: int) -> Dict[str, Dict[str, Any]]:
        if self._route_cache is not None and self._route_cache[0] == bar_idx:
            return self._route_cache[1]
        assets = sorted(set(self.holdings.keys()) | set(getattr(graph, "nodes", set())))
        if self.value_asset not in assets:
            assets.append(self.value_asset)
        route_quotes: Dict[str, Dict[str, Any]] = {
            asset: {"value": 0.0, "path": [], "edges": [], "hops": 0} for asset in assets
        }
        route_quotes[self.value_asset] = {"value": 1.0, "path": [self.value_asset], "edges": [], "hops": 0}
        if bar_idx < 0 or bar_idx >= len(getattr(graph, "common_timestamps", [])):
            self._route_cache = (bar_idx, route_quotes)
            return route_quotes

        edges_with_prices: List[Tuple[EdgeKey, str, str, float]] = []
        for edge in _canonical_graph_edges(graph):
            price = self.edge_price(graph, edge, bar_idx)
            if price <= 0.0:
                continue
            _, base_asset, quote_asset = _parse_edge(edge)
            discounted_price = price * math.exp(-max(0.0, float(getattr(graph, "fee_rate", 0.0))))
            edges_with_prices.append((edge, base_asset, quote_asset, discounted_price))

        for _ in range(max(0, len(assets) - 1)):
            changed = False
            for edge, base_asset, quote_asset, price in edges_with_prices:
                quote_value = route_quotes.get(quote_asset, {}).get("value", 0.0)
                if quote_value > 0.0:
                    candidate = price * quote_value
                    current = route_quotes.get(base_asset, {}).get("value", 0.0)
                    if candidate > current + 1e-12:
                        downstream_path = list(route_quotes.get(quote_asset, {}).get("path", []))
                        downstream_edges = list(route_quotes.get(quote_asset, {}).get("edges", []))
                        route_quotes[base_asset] = {
                            "value": candidate,
                            "path": [base_asset] + downstream_path,
                            "edges": [edge] + downstream_edges,
                            "hops": 1 + int(route_quotes.get(quote_asset, {}).get("hops", 0)),
                        }
                        changed = True
            if not changed:
                break

        self._route_cache = (bar_idx, route_quotes)
        return route_quotes

    def _route_values(self, graph: Any, bar_idx: int) -> Dict[str, float]:
        return {asset: float(info.get("value", 0.0)) for asset, info in self._route_quotes(graph, bar_idx).items()}

    def route_trace(self, graph: Any, bar_idx: int) -> Dict[str, Dict[str, Any]]:
        return self._route_quotes(graph, bar_idx)

    def worth(self, graph: Any, bar_idx: int) -> float:
        if self._worth_cache is not None and self._worth_cache[0] == bar_idx:
            return self._worth_cache[1]
        route_values = self._route_values(graph, bar_idx)
        worth = 0.0
        for asset, balance in self.holdings.items():
            worth += balance.total * route_values.get(asset, 0.0)
        self._worth_cache = (bar_idx, worth)
        return worth

    def by_value(self, graph: Any, bar_idx: int) -> List[Tuple[str, float]]:
        route_values = self._route_values(graph, bar_idx)
        ranked = []
        for asset, balance in self.holdings.items():
            if balance.total <= 0.0:
                continue
            ranked.append((asset, balance.total * route_values.get(asset, 0.0)))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def by_value_detailed(self, graph: Any, bar_idx: int) -> List[Dict[str, Any]]:
        quotes = self._route_quotes(graph, bar_idx)
        ranked: List[Dict[str, Any]] = []
        for asset, balance in self.holdings.items():
            if balance.total <= 0.0:
                continue
            quote = quotes.get(asset, {"value": 0.0, "path": [], "edges": [], "hops": 0})
            ranked.append(
                {
                    "asset": asset,
                    "qty": balance.total,
                    "value": balance.total * float(quote.get("value", 0.0)),
                    "route_value": float(quote.get("value", 0.0)),
                    "route_path": list(quote.get("path", [])),
                    "route_edges": list(quote.get("edges", [])),
                    "hops": int(quote.get("hops", 0)),
                }
            )
        return sorted(ranked, key=lambda item: item["value"], reverse=True)

    def snapshot(self, graph: Any, *, bar_idx: int, label: str = "snapshot") -> Dict[str, Any]:
        route_quotes = self._route_quotes(graph, bar_idx)
        snap = {
            "label": label,
            "bar_idx": bar_idx,
            "worth": self.worth(graph, bar_idx),
            "free_qty": self.free_qty_map(),
            "reserved_qty": self.reserved_qty_map(),
            "open_orders": self.open_order_count,
            "by_value": self.by_value(graph, bar_idx),
            "by_value_detailed": self.by_value_detailed(graph, bar_idx),
            "route_quotes": route_quotes,
        }
        self.snapshots.append(snap)
        return snap
