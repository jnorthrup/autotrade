"""
showdown.report – Leaderboard, equity-curve comparison, and JSON export
=========================================================================

Generates a formatted leaderboard table printed to console, ranking agents
by total return, Sharpe ratio, hit rate, and max drawdown.  Also produces
a side-by-side ASCII equity-curve comparison and per-agent breakdowns.

Public API
----------
print_leaderboard(agents)
    Accepts a list of agent-data dicts (see ``build_agent_reports``) or
    a ``ShowdownRunner`` instance.  Prints a rank table sorted by total
    return to stdout.

print_equity_curves(agents)
    Prints a simple ASCII side-by-side equity-curve summary for all agents.

print_per_agent_breakdown(agents)
    Prints detailed per-agent stats: final portfolio value, trade counts,
    winning / losing trades.

export_results(agents, filepath)
    Writes a JSON file with full per-agent metrics and trade history.

build_agent_reports(runner)
    Convenience helper: extract full report dicts from a ShowdownRunner.

Agent-data dict schema
----------------------
Each dict in the *agents* list is expected to contain at minimum:

    agent_name       : str
    initial_cash     : float
    final_value      : float
    total_pnl        : float
    return_pct       : float
    sharpe_estimate  : float | None
    hit_rate         : float
    trade_count      : int
    max_drawdown     : float
    max_drawdown_pct : float
    equity_curve     : list[float]
    trade_history    : list[dict]
    winning_trades   : int
    losing_trades    : int

Missing optional keys are filled with safe defaults.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np


# =====================================================================
# Helpers
# =====================================================================

def _normalise_agents(
    agents: Union[List[Dict[str, Any]], Any],
) -> List[Dict[str, Any]]:
    """
    Coerce the input into a list of agent-data dicts.

    Accepts:
      * A list of dicts (returned directly)
      * A ``ShowdownRunner`` instance (extracted via ``build_agent_reports``)
    """
    if isinstance(agents, list):
        return agents
    # Duck-type: if it has .agents and ._metrics it is probably a runner
    if hasattr(agents, "agents") and hasattr(agents, "_metrics"):
        return build_agent_reports(agents)
    raise TypeError(
        f"agents must be a list of dicts or a ShowdownRunner, got {type(agents)!r}"
    )


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Return d[key] if present and not None, else default."""
    val = d.get(key, default)
    return val if val is not None else default


def _compute_winning_losing(trade_history: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Walk the trade history using a FIFO cost-basis to count winning and
    losing round-trip trades.  Returns {"winning": int, "losing": int}.
    """
    from collections import defaultdict

    lots: Dict[str, List[List[float]]] = defaultdict(list)
    winning = 0
    losing = 0

    for trade in trade_history:
        action = trade.get("action", "")
        pair = trade.get("pair", "")
        size = float(trade.get("size", 0.0))
        price = float(trade.get("price", 0.0))

        if action == "BUY" and size > 0:
            lots[pair].append([size, price])
        elif action == "SELL" and size > 0:
            remaining = size
            trade_pnl = 0.0
            pair_lots = lots.get(pair, [])
            while remaining > 1e-15 and pair_lots:
                lot_qty, lot_price = pair_lots[0]
                filled = min(remaining, lot_qty)
                trade_pnl += (price - lot_price) * filled
                remaining -= filled
                lot_qty -= filled
                if lot_qty < 1e-15:
                    pair_lots.pop(0)
                else:
                    pair_lots[0] = [lot_qty, lot_price]
            if trade_pnl > 0:
                winning += 1
            elif trade_pnl < 0:
                losing += 1

    return {"winning": winning, "losing": losing}


# =====================================================================
# Build reports from runner
# =====================================================================

def build_agent_reports(runner: Any) -> List[Dict[str, Any]]:
    """
    Extract a list of agent-data dicts from a ``ShowdownRunner`` instance.

    Combines the ``Agent`` objects (for trade history) with the
    ``AgentMetrics`` objects (for computed metrics and equity curves).
    """
    reports: List[Dict[str, Any]] = []

    for name, agent in runner.agents.items():
        metrics = runner._metrics.get(name)
        if metrics is None:
            continue

        summary = metrics.compute_summary()
        trade_history = list(agent.trade_history)
        wl = _compute_winning_losing(trade_history)

        report = {
            "agent_name": name,
            "initial_cash": summary.get("initial_cash", 0.0),
            "final_value": summary.get("final_value", 0.0),
            "total_pnl": summary.get("total_pnl", 0.0),
            "return_pct": summary.get("return_pct", 0.0),
            "sharpe_estimate": summary.get("sharpe_estimate", 0.0),
            "hit_rate": summary.get("hit_rate", 0.0),
            "trade_count": summary.get("trade_count", 0),
            "max_drawdown": summary.get("max_drawdown", 0.0),
            "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
            "equity_curve": list(metrics.equity_curve),
            "trade_history": trade_history,
            "winning_trades": wl["winning"],
            "losing_trades": wl["losing"],
            "ticks_processed": summary.get("ticks_processed", 0),
        }
        reports.append(report)

    return reports


# =====================================================================
# Ranking helpers
# =====================================================================

def _rank_agents(agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort agents by ``return_pct`` descending and assign dense ranks.
    Agents with identical return_pct share the same rank.
    """
    # Sort descending by return_pct, break ties by agent_name for stability
    sorted_agents = sorted(
        agents,
        key=lambda a: (a.get("return_pct", 0.0), a.get("agent_name", "")),
        reverse=True,
    )

    # Assign dense ranks (ties get same rank)
    rank = 1
    for i, agent in enumerate(sorted_agents):
        if i > 0:
            prev_ret = sorted_agents[i - 1].get("return_pct", 0.0)
            curr_ret = agent.get("return_pct", 0.0)
            if curr_ret != prev_ret:
                rank = i + 1
        agent["_rank"] = rank

    return sorted_agents


# =====================================================================
# Formatted leaderboard
# =====================================================================

def print_leaderboard(
    agents: Union[List[Dict[str, Any]], Any],
    file: Optional[Any] = None,
) -> None:
    """
    Print a formatted leaderboard table sorted by total return.

    Columns: Rank, Agent, Return%, Sharpe, HitRate, Trades, MaxDD

    Edge cases:
      - Agent with zero trades shows "N/A" for Sharpe
      - Ties in return share the same rank
    """
    out = file or sys.stdout
    data = _normalise_agents(agents)
    ranked = _rank_agents(data)

    hdr = (
        f"{'Rank':>4}  {'Agent':<25} {'Return%':>9} "
        f"{'Sharpe':>8} {'HitRate':>8} {'Trades':>7} {'MaxDD':>8}"
    )
    sep = "=" * 80

    out.write("\n" + sep + "\n")
    out.write("LEADERBOARD (sorted by total return)\n")
    out.write(sep + "\n")
    out.write(hdr + "\n")
    out.write("-" * 80 + "\n")

    for a in ranked:
        rank_str = str(a["_rank"])
        name = a.get("agent_name", "unknown")[:25]
        ret_pct = a.get("return_pct", 0.0)

        # Sharpe: N/A for zero-trade agents
        trade_count = a.get("trade_count", 0)
        if trade_count == 0:
            sharpe_str = "N/A"
        else:
            sharpe_val = a.get("sharpe_estimate", 0.0)
            if sharpe_val is None:
                sharpe_str = "N/A"
            else:
                sharpe_str = f"{sharpe_val:>8.3f}"

        hit_rate = a.get("hit_rate", 0.0)
        max_dd = a.get("max_drawdown_pct", 0.0)

        out.write(
            f"{rank_str:>4}  {name:<25} "
            f"{ret_pct:>8.2f}% "
            f"{sharpe_str:>8} "
            f"{hit_rate:>7.2%} "
            f"{trade_count:>7d} "
            f"{max_dd:>7.2%}\n"
        )

    out.write(sep + "\n\n")


# =====================================================================
# Equity-curve ASCII comparison
# =====================================================================

def print_equity_curves(
    agents: Union[List[Dict[str, Any]], Any],
    num_bins: int = 20,
    width: int = 50,
    file: Optional[Any] = None,
) -> None:
    """
    Print a side-by-side ASCII equity-curve comparison for all agents.

    Each agent's equity curve is down-sampled to ``num_bins`` points and
    rendered as a horizontal sparkline (bar chart) using block characters.
    Summary stats (start, end, min, max, final return) are printed alongside.
    """
    out = file or sys.stdout
    data = _normalise_agents(agents)

    sep = "=" * 80
    out.write(sep + "\n")
    out.write("EQUITY CURVE COMPARISON\n")
    out.write(sep + "\n")

    for a in data:
        name = a.get("agent_name", "unknown")
        curve = a.get("equity_curve", [])
        initial = a.get("initial_cash", 0.0)
        ret_pct = a.get("return_pct", 0.0)

        if not curve or len(curve) < 2:
            out.write(f"  {name:<25}  (no equity data)\n")
            continue

        arr = np.asarray(curve, dtype=np.float64)

        # Down-sample
        if len(arr) > num_bins:
            indices = np.linspace(0, len(arr) - 1, num_bins, dtype=int)
            sampled = arr[indices]
        else:
            sampled = arr

        lo = float(np.min(sampled))
        hi = float(np.max(sampled))
        rng = hi - lo if hi != lo else 1.0

        # Build ASCII bar chart
        bar_chars = []
        for v in sampled:
            frac = (v - lo) / rng
            filled = int(frac * width)
            bar_chars.append("#" * filled + "-" * (width - filled))

        out.write(f"  {name:<25}  Return: {ret_pct:>8.2f}%\n")
        out.write(f"  {'Start':>8}: ${curve[0]:>14,.2f}   "
                  f"{'End':>8}: ${curve[-1]:>14,.2f}\n")
        out.write(f"  {'Min':>8}: ${lo:>14,.2f}   "
                  f"{'Max':>8}: ${hi:>14,.2f}\n")

        # Print a compact sparkline
        spark = _sparkline(sampled, width=width)
        out.write(f"  Spark: |{spark}|\n\n")

    out.write(sep + "\n\n")


def _sparkline(values: np.ndarray, width: int = 50) -> str:
    """
    Render a 1-line ASCII sparkline.  Maps *values* into a string of
    length *width* using characters ' ' '.' ':' '|' '#' to indicate level.
    """
    n = len(values)
    if n == 0:
        return " " * width

    lo = float(np.min(values))
    hi = float(np.max(values))
    rng = hi - lo if hi != lo else 1.0

    levels = " .:|#"  # 0-4
    chars = []
    for i in range(width):
        idx = int(i / width * n)
        idx = min(idx, n - 1)
        frac = (values[idx] - lo) / rng
        li = min(int(frac * (len(levels) - 1) + 0.5), len(levels) - 1)
        chars.append(levels[li])

    return "".join(chars)


# =====================================================================
# Per-agent breakdown
# =====================================================================

def print_per_agent_breakdown(
    agents: Union[List[Dict[str, Any]], Any],
    file: Optional[Any] = None,
) -> None:
    """
    Print a detailed per-agent breakdown: final portfolio value,
    number of trades, winning trades, losing trades.
    """
    out = file or sys.stdout
    data = _normalise_agents(agents)

    sep = "=" * 60

    out.write("\n" + sep + "\n")
    out.write("PER-AGENT BREAKDOWN\n")
    out.write(sep + "\n")

    for a in data:
        name = a.get("agent_name", "unknown")
        initial = a.get("initial_cash", 0.0)
        final = a.get("final_value", 0.0)
        pnl = a.get("total_pnl", 0.0)
        ret = a.get("return_pct", 0.0)
        trades = a.get("trade_count", 0)
        wins = a.get("winning_trades", 0)
        losses = a.get("losing_trades", 0)
        hit = a.get("hit_rate", 0.0)
        sharpe = a.get("sharpe_estimate", None)
        max_dd = a.get("max_drawdown_pct", 0.0)
        ticks = a.get("ticks_processed", 0)

        out.write(f"\n  Agent: {name}\n")
        out.write(f"    Initial Cash:    ${initial:>14,.2f}\n")
        out.write(f"    Final Value:     ${final:>14,.2f}\n")
        out.write(f"    Total P&L:       ${pnl:>14,.2f}\n")
        out.write(f"    Return:          {ret:>13.2f}%\n")

        if trades == 0:
            out.write("    Sharpe:                  N/A\n")
        else:
            s_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            out.write(f"    Sharpe:          {s_str:>13}\n")

        out.write(f"    Hit Rate:        {hit:>12.2%}\n")
        out.write(f"    Max Drawdown:    {max_dd:>12.2%}\n")
        out.write(f"    Total Trades:    {trades:>13d}\n")
        out.write(f"    Winning Trades:  {wins:>13d}\n")
        out.write(f"    Losing Trades:   {losses:>13d}\n")
        out.write(f"    Ticks Processed: {ticks:>13d}\n")

    out.write("\n" + sep + "\n\n")


# =====================================================================
# Overall summary
# =====================================================================

def print_overall_summary(
    agents: Union[List[Dict[str, Any]], Any],
    file: Optional[Any] = None,
) -> None:
    """
    Print an overall summary across all agents: who won, averages, etc.
    """
    out = file or sys.stdout
    data = _normalise_agents(agents)

    if not data:
        out.write("No agents to summarise.\n")
        return

    ranked = _rank_agents(data)

    winner = ranked[0]
    returns = [a.get("return_pct", 0.0) for a in data]
    avg_ret = float(np.mean(returns))
    med_ret = float(np.median(returns))
    total_trades = sum(a.get("trade_count", 0) for a in data)

    sep = "=" * 60
    out.write(sep + "\n")
    out.write("OVERALL SUMMARY\n")
    out.write(sep + "\n")
    out.write(f"  Agents:           {len(data):>10d}\n")
    out.write(f"  Total Trades:     {total_trades:>10d}\n")
    out.write(f"  Avg Return:       {avg_ret:>9.2f}%\n")
    out.write(f"  Median Return:    {med_ret:>9.2f}%\n")
    out.write(
        f"  Best:  {winner.get('agent_name', '?')} "
        f"({winner.get('return_pct', 0.0):.2f}%)\n"
    )
    worst = ranked[-1]
    out.write(
        f"  Worst: {worst.get('agent_name', '?')} "
        f"({worst.get('return_pct', 0.0):.2f}%)\n"
    )
    out.write(sep + "\n\n")


# =====================================================================
# JSON export
# =====================================================================

def export_results(
    agents: Union[List[Dict[str, Any]], Any],
    filepath: str,
) -> str:
    """
    Write a JSON file with full per-agent metrics and trade history.

    The output JSON structure::

        {
            "summary": { ...overall stats... },
            "agents": [
                {
                    "agent_name": "...",
                    "initial_cash": ...,
                    "final_value": ...,
                    "total_pnl": ...,
                    "return_pct": ...,
                    "sharpe_estimate": ...,
                    "hit_rate": ...,
                    "trade_count": ...,
                    "max_drawdown": ...,
                    "max_drawdown_pct": ...,
                    "winning_trades": ...,
                    "losing_trades": ...,
                    "ticks_processed": ...,
                    "equity_curve": [...],
                    "trade_history": [...]
                },
                ...
            ]
        }

    Returns the *filepath* that was written.
    """
    data = _normalise_agents(agents)
    ranked = _rank_agents(data)

    # Build clean agent records (strip internal _rank)
    agent_records = []
    for a in ranked:
        rec = {
            "rank": a["_rank"],
            "agent_name": a.get("agent_name", "unknown"),
            "initial_cash": a.get("initial_cash", 0.0),
            "final_value": a.get("final_value", 0.0),
            "total_pnl": a.get("total_pnl", 0.0),
            "return_pct": a.get("return_pct", 0.0),
            "sharpe_estimate": a.get("sharpe_estimate", None),
            "hit_rate": a.get("hit_rate", 0.0),
            "trade_count": a.get("trade_count", 0),
            "max_drawdown": a.get("max_drawdown", 0.0),
            "max_drawdown_pct": a.get("max_drawdown_pct", 0.0),
            "winning_trades": a.get("winning_trades", 0),
            "losing_trades": a.get("losing_trades", 0),
            "ticks_processed": a.get("ticks_processed", 0),
            "equity_curve": a.get("equity_curve", []),
            "trade_history": a.get("trade_history", []),
        }
        agent_records.append(rec)

    # Overall summary
    returns = [a.get("return_pct", 0.0) for a in data]
    overall = {
        "num_agents": len(data),
        "total_trades": sum(a.get("trade_count", 0) for a in data),
        "avg_return_pct": float(np.mean(returns)) if returns else 0.0,
        "median_return_pct": float(np.median(returns)) if returns else 0.0,
        "best_agent": ranked[0].get("agent_name", "") if ranked else None,
        "best_return_pct": ranked[0].get("return_pct", 0.0) if ranked else 0.0,
        "worst_agent": ranked[-1].get("agent_name", "") if ranked else None,
        "worst_return_pct": ranked[-1].get("return_pct", 0.0) if ranked else 0.0,
    }

    payload = {
        "summary": overall,
        "agents": agent_records,
    }

    with open(filepath, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_serial)

    return filepath


def _json_serial(obj: Any) -> Any:
    """Fallback serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
