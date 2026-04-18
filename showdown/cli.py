"""
showdown.cli – Command-line entry point for the showdown harness
=================================================================

Usage examples
--------------
  # Demo mode (simulated random-walk prices):
  python -m showdown.cli --agents 01,14,15,13,03 --ticks 200 --balance 10000 --demo

  # Historical replay from CSV:
  python -m showdown.cli --agents 01,14,15,13,03 --data prices.csv --balance 50000

  # Realtime mode (xtrade Java subprocess):
  python -m showdown.cli --agents 01,14,15,13,03 --pairs BTC/USDT,ETH/USDT --balance 10000 --ticks 500

Accepts command-line arguments:
  --agents   : comma-separated codec IDs (01-24), default 01,14,15,13,03,02
  --pairs    : comma-separated trading pairs, default BTC/USDT
  --ticks    : number of simulation cycles, default 200
  --balance  : starting cash per agent, default 10000
  --data     : optional price history CSV file for replay
  --demo     : flag to use simulated random-walk prices (identical to xtrade demo mode)
  --seed     : random seed for demo mode, default 42
  --verbose  : print progress every 10 ticks
  --export   : optional path to export results as JSON
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from .runner import ShowdownRunner, SimulatedDataSource
from .report import (
    build_agent_reports,
    print_leaderboard,
    print_equity_curves,
    print_per_agent_breakdown,
    print_overall_summary,
    export_results,
)


# ── Codec ID mapping (human-readable names) ──────────────────────────────

CODEC_NAMES: Dict[int, str] = {
    1: "volatility_breakout",
    2: "momentum_trend",
    3: "mean_reversion",
    4: "trend_following",
    5: "pairs_trading",
    6: "grid_trading",
    7: "volume_profile",
    8: "order_flow",
    9: "correlation_trading",
    10: "liquidity_making",
    11: "sector_rotation",
    12: "composite_alpha",
    13: "rsi_reversal",
    14: "bollinger_bands",
    15: "macd_crossover",
    16: "stochastic_kd",
    17: "adx_trend_strength",
    18: "vwap_mean_reversion",
    19: "kalman_filter_trend",
    20: "hurst_regime",
    21: "random_forest_classifier",
    22: "xgboost_signal",
    23: "transformer_attention",
    24: "zscore_stat_arb",
}

DEFAULT_AGENTS = [1, 14, 15, 13, 3, 2]
DEFAULT_PAIRS = ["BTC/USDT"]
DEFAULT_TICKS = 200
DEFAULT_BALANCE = 10000.0


def parse_agent_ids(agent_str: str) -> List[int]:
    """
    Parse a comma-separated string of codec IDs into a list of ints.

    Accepts formats like '01,14,15,13,03' or '1,14,15,13,3'.
    """
    ids = []
    for token in agent_str.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
        except ValueError:
            print(f"WARNING: Skipping invalid agent ID '{token}'", file=sys.stderr)
            continue
        if val < 1 or val > 24:
            print(
                f"WARNING: Agent ID {val} out of range (1-24), skipping",
                file=sys.stderr,
            )
            continue
        ids.append(val)
    return ids


def build_argparser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="showdown.cli",
        description="Showdown – multi-agent codec trading simulation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m showdown.cli --agents 01,14,15,13,03 --ticks 200 --balance 10000 --demo
  python -m showdown.cli --data prices.csv --balance 50000
  python -m showdown.cli --agents 01,02,03 --pairs BTC/USDT,ETH/USDT --ticks 500 --demo
        """,
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help=(
            "Comma-separated codec IDs (01-24). "
            "Default: 01,14,15,13,03,02 (volatility_breakout, bollinger_bands, "
            "macd_crossover, rsi_reversal, mean_reversion, momentum_trend)"
        ),
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated trading pairs (default: BTC/USDT)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_TICKS,
        help=f"Number of simulation cycles (default: {DEFAULT_TICKS})",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=DEFAULT_BALANCE,
        help=f"Starting cash per agent (default: {DEFAULT_BALANCE})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to price history CSV file for replay (columns: timestamp,pair,price,volume)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Use simulated random-walk prices (identical to xtrade's demo mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for demo mode (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress every 10 ticks",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results to a JSON file at the given path",
    )
    return parser


def run_showdown(args: argparse.Namespace) -> int:
    """
    Execute the showdown simulation based on parsed CLI arguments.

    Returns the exit code (0 for success, non-zero for error).
    """
    # ── Resolve agent IDs ────────────────────────────────────────────────
    if args.agents:
        codec_ids = parse_agent_ids(args.agents)
    else:
        codec_ids = list(DEFAULT_AGENTS)

    if not codec_ids:
        print("ERROR: No valid agent IDs provided.", file=sys.stderr)
        return 1

    # ── Resolve pairs ────────────────────────────────────────────────────
    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    else:
        pairs = list(DEFAULT_PAIRS)

    if not pairs:
        print("ERROR: No valid trading pairs provided.", file=sys.stderr)
        return 1

    # ── Determine data source ────────────────────────────────────────────
    if args.data:
        data_source = args.data
        source_label = f"replay: {args.data}"
    elif args.demo:
        data_source = {
            "type": "simulated",
            "pairs": pairs,
            "num_ticks": args.ticks,
            "seed": args.seed,
            "drift": 0.0001,
            "volatility": 0.02,
        }
        source_label = "demo (simulated random-walk)"
    else:
        # Default: if neither --data nor --demo, use demo mode
        data_source = {
            "type": "simulated",
            "pairs": pairs,
            "num_ticks": args.ticks,
            "seed": args.seed,
            "drift": 0.0001,
            "volatility": 0.02,
        }
        source_label = "demo (simulated random-walk)"

    # ── Print banner ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  S H O W D O W N   -   Multi-Agent Codec Trading Simulation")
    print("=" * 70)
    print(f"  Agents:     {len(codec_ids)} codecs")
    for cid in codec_ids:
        name = CODEC_NAMES.get(cid, f"codec_{cid:02d}")
        print(f"              [{cid:02d}] {name}")
    print(f"  Pairs:      {', '.join(pairs)}")
    print(f"  Ticks:      {args.ticks}")
    print(f"  Balance:    ${args.balance:,.2f} per agent")
    print(f"  Data:       {source_label}")
    if args.data:
        print(f"  Seed:       N/A (replay)")
    else:
        print(f"  Seed:       {args.seed}")
    print("=" * 70)
    print()

    # ── Build and run the runner ─────────────────────────────────────────
    try:
        runner = ShowdownRunner(
            codec_ids=codec_ids,
            data_source=data_source,
            initial_cash=args.balance,
            pairs=pairs,
            num_ticks=args.ticks,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialise runner: {e}", file=sys.stderr)
        return 1

    t0 = time.time()
    try:
        runner.run(num_ticks=args.ticks, verbose=args.verbose)
    except Exception as e:
        print(f"ERROR: Simulation failed: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0

    print(f"  Simulation completed in {elapsed:.2f}s")

    # ── Build reports and print output ───────────────────────────────────
    reports = build_agent_reports(runner)

    print_leaderboard(reports)
    print_equity_curves(reports)
    print_per_agent_breakdown(reports)
    print_overall_summary(reports)

    # ── Optional JSON export ─────────────────────────────────────────────
    if args.export:
        try:
            export_results(reports, args.export)
            print(f"  Results exported to: {args.export}")
        except Exception as e:
            print(f"  WARNING: Could not export results: {e}", file=sys.stderr)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point. Parses args and runs the showdown."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run_showdown(args)


if __name__ == "__main__":
    sys.exit(main())
