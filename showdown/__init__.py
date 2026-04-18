"""
showdown – unified indicator computation layer + agent adapter + runner + report
"""

from .indicators import compute_market_data, IndicatorComputer
from .agent import Agent, build_indicator_vec, ACTION_BUY, ACTION_SELL, ACTION_HOLD
from .runner import (
    ShowdownRunner,
    SimulatedDataSource,
    ReplayDataSource,
    RealtimeDataSource,
    AgentMetrics,
)
from .report import (
    print_leaderboard,
    print_equity_curves,
    print_per_agent_breakdown,
    print_overall_summary,
    export_results,
    build_agent_reports,
)

__all__ = [
    "compute_market_data",
    "IndicatorComputer",
    "Agent",
    "build_indicator_vec",
    "ACTION_BUY",
    "ACTION_SELL",
    "ACTION_HOLD",
    "ShowdownRunner",
    "SimulatedDataSource",
    "ReplayDataSource",
    "RealtimeDataSource",
    "AgentMetrics",
    "print_leaderboard",
    "print_equity_curves",
    "print_per_agent_breakdown",
    "print_overall_summary",
    "export_results",
    "build_agent_reports",
]
