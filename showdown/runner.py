"""
showdown.runner - Multi-agent showdown orchestrator
====================================================

Instantiates N codec agents from a configurable list of codec IDs,
feeds them identical tick data each cycle, collects their trade actions,
executes them against each agent's isolated paper portfolio, and tracks
per-agent performance metrics (P&L, Sharpe estimate, hit rate, trade
count, max drawdown).

Data sources:
  - Simulated: random-walk price generation
  - Replay: CSV or Parquet file (columns: timestamp, pair, price, volume)
  - Realtime: polling via xtrade Java subprocess or XChange REST endpoint
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .agent import Agent, ACTION_BUY, ACTION_SELL, ACTION_HOLD
from codec_models.base_codec import ExpertFactory


# =====================================================================
# Data sources
# =====================================================================

class SimulatedDataSource:
    """
    Generate synthetic random-walk tick data for one or more pairs.

    Parameters
    ----------
    pairs : list[str]
        Trading pairs to simulate, e.g. ["BTC/USDT", "ETH/USDT"].
    num_ticks : int
        Total ticks to produce.
    base_prices : dict[str, float], optional
        Starting price per pair (default 100.0).
    seed : int
        Random seed for reproducibility.
    drift : float
        Mean log-return per tick.
    volatility : float
        Std-dev of log-return per tick.
    """

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        num_ticks: int = 100,
        base_prices: Optional[Dict[str, float]] = None,
        seed: int = 42,
        drift: float = 0.0001,
        volatility: float = 0.02,
    ) -> None:
        self.pairs = pairs or ["BTC/USDT"]
        self.num_ticks = num_ticks
        self.base_prices = base_prices or {p: 100.0 for p in self.pairs}
        self.seed = seed
        self.drift = drift
        self.volatility = volatility

        self._rng: np.random.RandomState = np.random.RandomState(seed)
        self._tick_idx: int = 0
        self._prices: Dict[str, float] = dict(self.base_prices)
        self._epoch_start: float = 1_700_000_000.0

    # -- iterator protocol ------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Dict[str, Any]]:
        if self._tick_idx >= self.num_ticks:
            raise StopIteration

        tickers: Dict[str, Dict[str, Any]] = {}
        for pair in self.pairs:
            log_ret = self._rng.normal(self.drift, self.volatility)
            self._prices[pair] *= np.exp(log_ret)
            vol = float(self._rng.uniform(100.0, 10_000.0))
            tickers[pair] = {
                "price": float(self._prices[pair]),
                "volume": vol,
            }
        self._tick_idx += 1
        return tickers

    def reset(self) -> None:
        self._rng = np.random.RandomState(self.seed)
        self._tick_idx = 0
        self._prices = dict(self.base_prices)


class ReplayDataSource:
    """
    Replay tick data from a CSV or Parquet file.

    Expected columns: timestamp, pair, price, volume

    Consecutive rows sharing the same *timestamp* are grouped into one
    tick (so multiple pairs can appear per tick).

    Parameters
    ----------
    filepath : str
        Path to CSV (.csv) or Parquet (.parquet) file.
    max_ticks : int | None
        Maximum number of *ticks* (unique timestamps) to yield.
    """

    def __init__(
        self,
        filepath: str,
        max_ticks: Optional[int] = None,
    ) -> None:
        self.filepath = filepath
        self.max_ticks = max_ticks
        self._rows: List[Dict[str, Any]] = []
        self._tick_idx: int = 0
        self._len: int = 0
        self._load()

    # -- loading ----------------------------------------------------------

    def _load(self) -> None:
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext == ".parquet":
            self._load_parquet()
        else:
            self._load_csv()
        # Pre-group rows by timestamp to know total tick count
        self._groups: List[List[Dict[str, Any]]] = []
        if self._rows:
            current_ts = self._rows[0]["timestamp"]
            group: List[Dict[str, Any]] = []
            for row in self._rows:
                if row["timestamp"] != current_ts:
                    self._groups.append(group)
                    group = []
                    current_ts = row["timestamp"]
                group.append(row)
            if group:
                self._groups.append(group)
        self._len = len(self._groups)

    def _load_csv(self) -> None:
        with open(self.filepath, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._rows.append({
                    "timestamp": float(row["timestamp"]),
                    "pair": str(row["pair"]).strip(),
                    "price": float(row["price"]),
                    "volume": float(row["volume"]),
                })

    def _load_parquet(self) -> None:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas + pyarrow are required to read Parquet files"
            ) from exc
        df = pd.read_parquet(self.filepath)
        for _, row in df.iterrows():
            self._rows.append({
                "timestamp": float(row["timestamp"]),
                "pair": str(row["pair"]),
                "price": float(row["price"]),
                "volume": float(row["volume"]),
            })

    # -- iterator protocol ------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Dict[str, Any]]:
        if self.max_ticks is not None and self._tick_idx >= self.max_ticks:
            raise StopIteration
        if self._tick_idx >= self._len:
            raise StopIteration

        group = self._groups[self._tick_idx]
        tickers: Dict[str, Dict[str, Any]] = {}
        for row in group:
            tickers[row["pair"]] = {
                "price": row["price"],
                "volume": row["volume"],
            }
        self._tick_idx += 1
        return tickers

    def __len__(self) -> int:
        return self._len

    def reset(self) -> None:
        self._tick_idx = 0


class RealtimeDataSource:
    """
    Poll live market prices via an xtrade Java subprocess or an
    XChange-compatible REST API.

    Parameters
    ----------
    pairs : list[str]
        Trading pairs to poll.
    mode : str
        ``"xtrade"`` or ``"xchange_rest"``.
    poll_interval : float
        Seconds between polls.
    subprocess_cmd : list[str] | None
        Command vector to start the xtrade subprocess.
    rest_endpoint : str | None
        Base URL of an XChange-compatible REST ticker endpoint.
    max_ticks : int | None
        Stop after this many ticks (None = unlimited).
    """

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        mode: str = "xchange_rest",
        poll_interval: float = 1.0,
        subprocess_cmd: Optional[List[str]] = None,
        rest_endpoint: Optional[str] = None,
        max_ticks: Optional[int] = None,
    ) -> None:
        self.pairs = pairs or ["BTC/USDT"]
        self.mode = mode
        self.poll_interval = poll_interval
        self.subprocess_cmd = subprocess_cmd or ["java", "-jar", "xtrade.jar"]
        self.rest_endpoint = rest_endpoint
        self.max_ticks = max_ticks
        self._tick_idx: int = 0
        self._process: Optional[subprocess.Popen] = None
        self._fallback_rng = np.random.RandomState(99)
        self._fallback_prices: Dict[str, float] = {p: 100.0 for p in self.pairs}

    # -- iterator protocol ------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Dict[str, Any]]:
        if self.max_ticks is not None and self._tick_idx >= self.max_ticks:
            raise StopIteration

        if self.mode == "xtrade":
            result = self._poll_xtrade()
        elif self.mode == "xchange_rest":
            result = self._poll_rest()
        else:
            raise ValueError(f"Unknown realtime mode: {self.mode!r}")

        self._tick_idx += 1
        return result

    # -- xtrade subprocess ------------------------------------------------

    def _poll_xtrade(self) -> Dict[str, Dict[str, Any]]:
        if self._process is None:
            try:
                self._process = subprocess.Popen(
                    self.subprocess_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except (FileNotFoundError, OSError):
                return self._fallback()

        tickers: Dict[str, Dict[str, Any]] = {}
        try:
            line = self._process.stdout.readline()
            if not line:
                return self._fallback()
            payload = json.loads(line.strip())
            for pair in self.pairs:
                if pair in payload:
                    tickers[pair] = {
                        "price": float(payload[pair].get("price", 100.0)),
                        "volume": float(payload[pair].get("volume", 0.0)),
                    }
                else:
                    tickers[pair] = self._fallback_entry(pair)
        except (json.JSONDecodeError, KeyError, ValueError):
            tickers = {p: self._fallback_entry(p) for p in self.pairs}

        return tickers

    # -- REST polling -----------------------------------------------------

    def _poll_rest(self) -> Dict[str, Dict[str, Any]]:
        if not self.rest_endpoint:
            return self._fallback()

        tickers: Dict[str, Dict[str, Any]] = {}
        try:
            import urllib.request
            for pair in self.pairs:
                symbol = pair.replace("/", "")
                url = f"{self.rest_endpoint}/ticker/{symbol}"
                req = urllib.request.Request(
                    url, headers={"User-Agent": "autotrade/1.0"}
                )
                with urllib.request.urlopen(req, timeout=self.poll_interval) as resp:
                    data = json.loads(resp.read().decode())
                    tickers[pair] = {
                        "price": float(
                            data.get("last", data.get("price", 100.0))
                        ),
                        "volume": float(data.get("volume", 0.0)),
                    }
        except Exception:
            return self._fallback()
        return tickers

    # -- helpers ----------------------------------------------------------

    def _fallback(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for pair in self.pairs:
            lr = self._fallback_rng.normal(0.0, 0.005)
            self._fallback_prices[pair] *= np.exp(lr)
            out[pair] = {
                "price": float(self._fallback_prices[pair]),
                "volume": float(self._fallback_rng.uniform(100, 5000)),
            }
        return out

    def _fallback_entry(self, pair: str) -> Dict[str, Any]:
        lr = self._fallback_rng.normal(0.0, 0.005)
        self._fallback_prices[pair] *= np.exp(lr)
        return {
            "price": float(self._fallback_prices[pair]),
            "volume": float(self._fallback_rng.uniform(100, 5000)),
        }

    def stop(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process = None

    def reset(self) -> None:
        self._tick_idx = 0
        self.stop()


# =====================================================================
# Per-agent metrics tracker
# =====================================================================

class AgentMetrics:
    """
    Accumulates per-agent performance metrics tick-by-tick.

    Tracks equity curve, realized P&L (FIFO cost-basis), unrealized P&L,
    trade count, max drawdown, Sharpe estimate, and hit rate.
    """

    def __init__(self, agent_name: str, initial_cash: float) -> None:
        self.agent_name = agent_name
        self.initial_cash = initial_cash

        # FIFO cost basis: pair -> list of (qty, price)
        self._cost_lots: Dict[str, List[List[float]]] = defaultdict(list)

        self.realized_pnl: float = 0.0
        self.equity_curve: List[float] = [initial_cash]
        self.snapshots: List[Dict[str, Any]] = []
        self.total_trades: int = 0
        self._winning_trades: int = 0
        self._round_trip_trades: int = 0

    # ------------------------------------------------------------------

    def record_tick(
        self,
        cash: float,
        holdings: Dict[str, float],
        prices: Dict[str, float],
        actions: List[Dict[str, Any]],
        tick: int,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Process one tick's actions and return the snapshot dict."""

        for act in actions:
            pair = act["pair"]
            action = act["action"]
            size = act["size"]
            price = act["price"]

            if action == ACTION_BUY and size > 0:
                self._cost_lots[pair].append([size, price])
                self.total_trades += 1

            elif action == ACTION_SELL and size > 0:
                remaining = size
                trade_pnl = 0.0
                lots = self._cost_lots.get(pair, [])
                while remaining > 1e-15 and lots:
                    lot_qty, lot_price = lots[0]
                    filled = min(remaining, lot_qty)
                    trade_pnl += (price - lot_price) * filled
                    remaining -= filled
                    lot_qty -= filled
                    if lot_qty < 1e-15:
                        lots.pop(0)
                    else:
                        lots[0] = [lot_qty, lot_price]

                self.realized_pnl += trade_pnl
                self.total_trades += 1
                self._round_trip_trades += 1
                if trade_pnl > 0:
                    self._winning_trades += 1

        # Mark-to-market
        holdings_value = sum(
            qty * prices.get(pair, 0.0)
            for pair, qty in holdings.items()
        )
        total_value = cash + holdings_value
        unrealized_pnl = total_value - self.initial_cash
        self.equity_curve.append(total_value)

        snapshot: Dict[str, Any] = {
            "tick": tick,
            "timestamp": timestamp,
            "cash": cash,
            "holdings_value": holdings_value,
            "total_value": total_value,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "trade_count": self.total_trades,
        }
        self.snapshots.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------

    def compute_summary(self) -> Dict[str, Any]:
        """Return aggregate metrics over the full recorded history."""
        final_value = self.equity_curve[-1] if self.equity_curve else self.initial_cash
        total_pnl = final_value - self.initial_cash
        return_pct = (
            (total_pnl / self.initial_cash) * 100.0
            if self.initial_cash
            else 0.0
        )

        # Sharpe estimate (annualised; 1 tick ~ 1 s -> 86400 ticks/day)
        eq = np.asarray(self.equity_curve, dtype=np.float64)
        sharpe = 0.0
        if eq.size > 1:
            rets = np.diff(eq) / eq[:-1]
            rets = rets[np.isfinite(rets)]
            if rets.size > 1 and np.std(rets) > 1e-12:
                sharpe = float(
                    np.mean(rets) / np.std(rets) * np.sqrt(86400 * 252)
                )

        # Max drawdown (absolute and percent)
        max_dd = 0.0
        max_dd_pct = 0.0
        if eq.size > 0:
            peak = eq[0]
            for v in eq:
                if v > peak:
                    peak = v
                dd = peak - v
                dd_pct = dd / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

        hit_rate = (
            self._winning_trades / self._round_trip_trades
            if self._round_trip_trades > 0
            else 0.0
        )

        return {
            "agent_name": self.agent_name,
            "initial_cash": self.initial_cash,
            "final_value": float(final_value),
            "total_pnl": float(total_pnl),
            "return_pct": float(return_pct),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(final_value - self.initial_cash - self.realized_pnl),
            "sharpe_estimate": float(sharpe),
            "hit_rate": float(hit_rate),
            "trade_count": self.total_trades,
            "max_drawdown": float(max_dd),
            "max_drawdown_pct": float(max_dd_pct),
            "ticks_processed": len(self.snapshots),
        }


# =====================================================================
# ShowdownRunner
# =====================================================================

class ShowdownRunner:
    """
    Multi-agent showdown orchestrator.

    Parameters
    ----------
    codec_ids : list[int]
        Codec expert IDs (1-24) to instantiate.
    data_source : str | dict | iterable
        * ``"simulated"`` - auto-generate random ticks
        * file path (``.csv`` / ``.parquet``) - replay from file
        * dict with ``"type"`` key (``"simulated"``, ``"replay"``,
          ``"realtime"``) and type-specific config
        * any iterable yielding ``{pair: {"price": float, "volume": float}}``
    initial_cash : float
        Starting paper-cash per agent (default 100 000).
    pairs : list[str]
        Trading pairs (used by simulated / realtime sources).
    num_ticks : int
        Tick budget for simulated mode.
    """

    def __init__(
        self,
        codec_ids: List[int],
        data_source: Union[str, Dict[str, Any], Any] = "simulated",
        initial_cash: float = 100_000.0,
        pairs: Optional[List[str]] = None,
        num_ticks: int = 100,
    ) -> None:
        self.codec_ids = list(codec_ids)
        self.initial_cash = initial_cash
        self.pairs = pairs or ["BTC/USDT"]
        self.num_ticks = num_ticks

        # Instantiate one Agent per codec ID
        self.agents: Dict[str, Agent] = {}
        self._metrics: Dict[str, AgentMetrics] = {}
        for cid in self.codec_ids:
            codec = ExpertFactory.create_expert(cid)
            agent = Agent(codec, initial_cash=initial_cash)
            name = codec.name
            self.agents[name] = agent
            self._metrics[name] = AgentMetrics(name, initial_cash)

        # Build data source
        self.data_source = self._build_data_source(data_source)

        # Runtime bookkeeping
        self._tick_count: int = 0
        self._last_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Data-source construction
    # ------------------------------------------------------------------

    def _build_data_source(
        self, data_source: Union[str, Dict[str, Any], Any]
    ) -> Any:
        if isinstance(data_source, str):
            if data_source == "simulated":
                return SimulatedDataSource(
                    pairs=self.pairs, num_ticks=self.num_ticks, seed=42,
                )
            if os.path.isfile(data_source):
                return ReplayDataSource(
                    data_source, max_ticks=self.num_ticks,
                )
            raise ValueError(
                f"Unknown data source string: {data_source!r}. "
                "Use 'simulated', a file path, or a config dict."
            )

        if isinstance(data_source, dict):
            ds_type = data_source.get("type", "simulated")
            if ds_type == "simulated":
                return SimulatedDataSource(
                    pairs=data_source.get("pairs", self.pairs),
                    num_ticks=data_source.get("num_ticks", self.num_ticks),
                    base_prices=data_source.get("base_prices"),
                    seed=data_source.get("seed", 42),
                    drift=data_source.get("drift", 0.0001),
                    volatility=data_source.get("volatility", 0.02),
                )
            if ds_type == "replay":
                return ReplayDataSource(
                    data_source["filepath"],
                    max_ticks=data_source.get("max_ticks", self.num_ticks),
                )
            if ds_type == "realtime":
                return RealtimeDataSource(
                    pairs=data_source.get("pairs", self.pairs),
                    mode=data_source.get("mode", "xchange_rest"),
                    poll_interval=data_source.get("poll_interval", 1.0),
                    subprocess_cmd=data_source.get("subprocess_cmd"),
                    rest_endpoint=data_source.get("rest_endpoint"),
                    max_ticks=data_source.get("max_ticks", self.num_ticks),
                )
            raise ValueError(f"Unknown data-source type: {ds_type!r}")

        # Assume it is already an iterable
        return data_source

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        num_ticks: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute the showdown.

        Parameters
        ----------
        num_ticks : int | None
            Override tick limit.  None = exhaust data source or use
            ``self.num_ticks``.
        verbose : bool
            Print a progress line every 10 ticks.

        Returns
        -------
        dict[str, dict]
            Per-agent summary metrics ``{agent_name: summary}``.
        """
        limit = num_ticks if num_ticks is not None else self.num_ticks
        self._tick_count = 0

        for tick_data in self.data_source:
            if self._tick_count >= limit:
                break

            ts = time.time()
            self._last_prices = {
                pair: td["price"] for pair, td in tick_data.items()
            }

            # Feed *identical* tick data to every agent
            for name, agent in self.agents.items():
                actions = agent.on_tick(tick_data)
                self._metrics[name].record_tick(
                    cash=agent.cash,
                    holdings=dict(agent.holdings),
                    prices=self._last_prices,
                    actions=actions,
                    tick=self._tick_count,
                    timestamp=ts,
                )

            self._tick_count += 1

            if verbose and self._tick_count % 10 == 0:
                parts = [
                    f"{n}=${m.equity_curve[-1]:,.0f}"
                    for n, m in self._metrics.items()
                ]
                print(f"  tick {self._tick_count}/{limit}: {', '.join(parts)}")

        return self.get_summary()

    # ------------------------------------------------------------------
    # Convenience: run from a CSV / Parquet file
    # ------------------------------------------------------------------

    def run_replay(
        self,
        filepath: str,
        num_ticks: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Replay a price-history file through all agents.

        Parameters
        ----------
        filepath : str
            CSV or Parquet file with columns: timestamp, pair, price, volume.
        num_ticks : int | None
            Max ticks to replay.

        Returns
        -------
        dict[str, dict]
            Per-agent summary metrics.
        """
        replay_ds = ReplayDataSource(filepath, max_ticks=num_ticks)
        saved_ds = self.data_source
        self.data_source = replay_ds
        self.reset()
        result = self.run(num_ticks=num_ticks)
        self.data_source = saved_ds
        return result

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Per-agent summary metrics."""
        return {
            name: metrics.compute_summary()
            for name, metrics in self._metrics.items()
        }

    def get_snapshots(self) -> Dict[str, List[Dict[str, Any]]]:
        """Per-agent tick-level snapshots."""
        return {
            name: list(metrics.snapshots)
            for name, metrics in self._metrics.items()
        }

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Agents ranked by total P&L (descending)."""
        summaries = self.get_summary()
        ranked = sorted(
            summaries.values(),
            key=lambda s: s["total_pnl"],
            reverse=True,
        )
        for i, entry in enumerate(ranked):
            entry["rank"] = i + 1
        return ranked

    def print_leaderboard(self) -> None:
        """Print a formatted leaderboard table to stdout."""
        lb = self.get_leaderboard()
        hdr = (
            f"{'Rank':>4}  {'Agent':<25} {'P&L':>12} {'Ret%':>9} "
            f"{'Sharpe':>8} {'HitRate':>8} {'Trades':>7} {'MaxDD%':>8}"
        )
        print("\n" + "=" * 90)
        print("SHOWDOWN LEADERBOARD")
        print("=" * 90)
        print(hdr)
        print("-" * 90)
        for e in lb:
            print(
                f"{e['rank']:>4}  {e['agent_name']:<25} "
                f"{e['total_pnl']:>12,.2f} {e['return_pct']:>8.2f}% "
                f"{e['sharpe_estimate']:>8.3f} {e['hit_rate']:>7.2%} "
                f"{e['trade_count']:>7d} {e['max_drawdown_pct']:>7.2%}"
            )
        print("=" * 90 + "\n")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset every agent and metric tracker to initial state."""
        for agent in self.agents.values():
            agent.reset()
        for name in self._metrics:
            self._metrics[name] = AgentMetrics(name, self.initial_cash)
        self._tick_count = 0
        self._last_prices = {}
        if hasattr(self.data_source, "reset"):
            self.data_source.reset()
