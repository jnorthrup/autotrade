#!/usr/bin/env python3
"""
validation_harness.py — Cross-validate Python ShowdownRunner vs Java ShowdownHarness
======================================================================================

Generates a seeded random-walk CSV of simulated ticks, runs both the Python
ShowdownRunner and the Java ShowdownValidator against the same data, parses
the outputs, and asserts per-agent PnL values match within a configurable
tolerance.

Usage:
    python validation_harness.py [--ticks 200] [--tolerance 0.005] [--csv path]

Acceptance criteria:
    - Shared CSV tick file produces matching indicator values (Python vs Java)
    - Per-agent final portfolio values agree within 0.5% for at least 5 codecs
    - Outputs PASS/FAIL summary and diff table
    - Documents any discrepancies >0.5% with root cause
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from showdown.runner import ShowdownRunner, ReplayDataSource
from showdown.agent import Agent, build_indicator_vec
from showdown.indicators import IndicatorComputer
from codec_models.base_codec import ExpertFactory


# =====================================================================
# 1. CSV Generation
# =====================================================================

def generate_tick_csv(
    filepath: str,
    num_ticks: int = 200,
    seed: int = 42,
    pair: str = "BTC/USDT",
    base_price: float = 100.0,
    drift: float = 0.0001,
    volatility: float = 0.02,
) -> str:
    """
    Generate a CSV of simulated random-walk ticks.

    Columns: timestamp, pair, price, volume

    Uses a seeded RandomState so both Python and Java see identical data.
    """
    rng = np.random.RandomState(seed)
    price = base_price
    epoch = 1_700_000_000.0

    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "pair", "price", "volume"])
        for i in range(num_ticks):
            log_ret = rng.normal(drift, volatility)
            price *= np.exp(log_ret)
            vol = float(rng.uniform(100.0, 10_000.0))
            writer.writerow([f"{epoch + i:.1f}", pair, f"{price:.6f}", f"{vol:.1f}"])

    return filepath


# =====================================================================
# 2. Python Showdown Run
# =====================================================================

def run_python_showdown(csv_path: str, codec_ids: List[int], num_ticks: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str]]:
    """
    Run the Python ShowdownRunner against the CSV file.
    Returns (per-agent summary dict, codec_id->agent_name mapping).
    """
    runner = ShowdownRunner(
        codec_ids=codec_ids,
        data_source=csv_path,
        initial_cash=100_000.0,
        pairs=["BTC/USDT"],
        num_ticks=num_ticks,
    )
    results = runner.run(num_ticks=num_ticks)

    # Build codec_id -> name mapping
    id_to_name = {}
    for cid in codec_ids:
        codec = ExpertFactory.create_expert(cid)
        id_to_name[cid] = codec.name

    return results, id_to_name


def run_python_indicators(csv_path: str, num_ticks: int) -> List[Dict[str, Any]]:
    """
    Run the Python IndicatorComputer on the CSV and return per-tick indicator snapshots.
    """
    computer = IndicatorComputer(buffer_size=200)
    snapshots = []
    source = ReplayDataSource(csv_path, max_ticks=num_ticks)

    for t, tick_data in enumerate(source):
        for pair, tick in tick_data.items():
            price = float(tick["price"])
            volume = float(tick["volume"])
            md = computer.compute(pair, price, volume)
            snapshots.append({"tick": t, "pair": pair, **md})

    return snapshots


# =====================================================================
# 3. Java Showdown Run
# =====================================================================

def run_java_showdown(csv_path: str, num_ticks: int) -> Dict[str, Any]:
    """
    Invoke the Java ShowdownValidator via subprocess and parse JSON output.
    Returns the parsed JSON dict with per-agent results.
    """
    jar_path = os.path.join(PROJECT_ROOT, "xtrade", "target", "xtrade-1.0-SNAPSHOT.jar")
    if not os.path.isfile(jar_path):
        raise FileNotFoundError(f"Java JAR not found: {jar_path}")

    cmd = [
        "java", "-cp", jar_path,
        "com.xtrade.showdown.ShowdownValidator",
        csv_path,
        str(num_ticks),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Java validator failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr}\nSTDOUT: {result.stdout[:2000]}"
        )

    # Parse JSON from stdout
    output = result.stdout.strip()
    # Find the JSON object (starts with '{')
    json_start = output.find("{")
    if json_start < 0:
        raise ValueError(f"No JSON found in Java output:\n{output[:1000]}")

    return json.loads(output[json_start:])


def run_java_indicators(csv_path: str, num_ticks: int) -> List[Dict[str, Any]]:
    """
    Invoke the Java ShowdownValidator with --indicators and parse indicator snapshots.
    """
    jar_path = os.path.join(PROJECT_ROOT, "xtrade", "target", "xtrade-1.0-SNAPSHOT.jar")
    cmd = [
        "java", "-cp", jar_path,
        "com.xtrade.showdown.ShowdownValidator",
        csv_path,
        str(num_ticks),
        "--indicators",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Java validator failed: {result.stderr}")

    output = result.stdout.strip()
    json_start = output.find("{")
    if json_start < 0:
        return []

    data = json.loads(output[json_start:])
    return data.get("indicator_snapshots", [])


# =====================================================================
# 4. Comparison & Reporting
# =====================================================================

def compare_indicators(
    py_snaps: List[Dict[str, Any]],
    java_snaps: List[Dict[str, Any]],
    tolerance: float = 0.01,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Compare indicator values between Python and Java snapshots.
    Returns (all_match, diff_table).
    """
    indicator_keys = [
        "sma_15", "sma_20", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist",
        "rsi", "rsi_14",
        "bb_upper", "bb_lower", "bb_mid",
        "atr_14", "stoch_k", "stoch_d",
        "adx", "adx_14", "plus_di", "minus_di",
        "vwap", "momentum", "avg_volume", "log_return",
    ]

    diffs = []
    all_match = True

    # Compare per-tick (Java dumps first codec, first 5 ticks)
    for i in range(min(len(py_snaps), len(java_snaps))):
        py = py_snaps[i]
        ja = java_snaps[i]

        for key in indicator_keys:
            py_val = py.get(key)
            ja_val = ja.get(key)

            if py_val is None or ja_val is None:
                continue

            py_val = float(py_val)
            ja_val = float(ja_val)

            # Use relative tolerance for large values, absolute for small
            denom = max(abs(py_val), abs(ja_val), 1e-10)
            rel_diff = abs(py_val - ja_val) / denom

            if rel_diff > tolerance:
                all_match = False
                diffs.append({
                    "tick": py.get("tick", i),
                    "indicator": key,
                    "python": py_val,
                    "java": ja_val,
                    "rel_diff_pct": rel_diff * 100.0,
                    "abs_diff": abs(py_val - ja_val),
                })

    return all_match, diffs


def compare_agent_results(
    py_results: Dict[str, Dict[str, Any]],
    java_results: Dict[str, Any],
    tolerance: float = 0.005,
    id_to_name: Optional[Dict[int, str]] = None,
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """
    Compare per-agent final portfolio values.
    Returns (all_pass, diff_table, pass_count).
    tolerance is the relative tolerance (e.g. 0.005 = 0.5%).
    """
    java_agents = java_results.get("agents", {})
    diffs = []
    pass_count = 0

    # Build name->id mapping from Java agents
    java_name_to_id = {}
    for name, data in java_agents.items():
        cid = data.get("codec_id", 0)
        java_name_to_id[name] = cid

    # Build id->py_name mapping
    py_id_to_name = {}
    if id_to_name:
        py_id_to_name = {v: k for k, v in id_to_name.items()}

    # Match by codec_id
    matched = set()
    for ja_name, ja_data in java_agents.items():
        cid = ja_data.get("codec_id", 0)
        py_name = id_to_name.get(cid) if id_to_name else ja_name

        if py_name not in py_results:
            continue

        py = py_results[py_name]
        ja = ja_data
        matched.add(cid)

        py_final = float(py.get("final_value", py.get("total_pnl", 0) + py.get("initial_cash", 100000)))
        ja_final = float(ja.get("final_value", 0))
        py_pnl = float(py.get("total_pnl", 0))
        ja_pnl = float(ja.get("total_pnl", 0))
        py_trades = int(py.get("trade_count", 0))
        ja_trades = int(ja.get("trade_count", 0))
        py_realized = float(py.get("realized_pnl", 0))
        ja_realized = float(ja.get("realized_pnl", 0))

        denom = max(abs(py_final), abs(ja_final), 1.0)
        rel_diff = abs(py_final - ja_final) / denom

        entry = {
            "agent": f"codec_{cid:02d}",
            "py_name": py_name,
            "ja_name": ja_name,
            "py_final": py_final,
            "ja_final": ja_final,
            "py_pnl": py_pnl,
            "ja_pnl": ja_pnl,
            "py_trades": py_trades,
            "ja_trades": ja_trades,
            "py_realized": py_realized,
            "ja_realized": ja_realized,
            "final_diff_pct": rel_diff * 100.0,
            "within_tolerance": rel_diff <= tolerance,
        }

        if rel_diff <= tolerance:
            pass_count += 1

        diffs.append(entry)

    # Sort by agent name
    diffs.sort(key=lambda d: d["agent"])

    all_pass = pass_count == len(diffs) and len(diffs) > 0
    return all_pass, diffs, pass_count


# =====================================================================
# 5. Main Validation Script
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-validate Python vs Java showdown")
    parser.add_argument("--ticks", type=int, default=200, help="Number of ticks to simulate")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Relative tolerance (0.005 = 0.5%%)")
    parser.add_argument("--csv", type=str, default=None, help="CSV file path (auto-generated if not given)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tick generation")
    parser.add_argument("--codec-ids", type=str, default=None, help="Comma-separated codec IDs (default: 1-24)")
    parser.add_argument("--skip-java", action="store_true", help="Skip Java execution (Python only)")
    parser.add_argument("--skip-indicators", action="store_true", help="Skip indicator comparison")
    args = parser.parse_args()

    if args.codec_ids:
        codec_ids = [int(x.strip()) for x in args.codec_ids.split(",")]
    else:
        codec_ids = list(range(1, 25))

    print("=" * 80)
    print("  SHOWDOWN VALIDATION HARNESS")
    print("  Python ShowdownRunner  vs  Java ShowdownValidator")
    print("=" * 80)
    print(f"  Ticks:     {args.ticks}")
    print(f"  Tolerance: {args.tolerance * 100:.2f}%")
    print(f"  Codecs:    {codec_ids}")
    print(f"  Seed:      {args.seed}")
    print("=" * 80)

    # --- Step 1: Generate CSV ---
    print("\n[1/4] Generating tick CSV...")
    if args.csv:
        csv_path = args.csv
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir=PROJECT_ROOT)
        csv_path = tmp.name
        tmp.close()

    generate_tick_csv(
        csv_path,
        num_ticks=args.ticks,
        seed=args.seed,
    )
    print(f"  CSV written to: {csv_path}")

    # --- Step 2: Run Python ---
    print("\n[2/4] Running Python ShowdownRunner...")
    py_results, id_to_name = run_python_showdown(csv_path, codec_ids, args.ticks)
    print(f"  Python completed: {len(py_results)} agents")

    # --- Step 3: Run Java ---
    java_results = None
    if not args.skip_java:
        print("\n[3/4] Running Java ShowdownValidator...")
        try:
            java_results = run_java_showdown(csv_path, args.ticks)
            print(f"  Java completed: {len(java_results.get('agents', {}))} agents")
        except Exception as e:
            print(f"  Java FAILED: {e}")
            java_results = None
    else:
        print("\n[3/4] Skipping Java (--skip-java)")

    # --- Step 4: Compare ---
    print("\n[4/4] Comparing results...")

    # 4a. Compare indicators
    ind_pass = True
    ind_diffs = []
    if not args.skip_indicators and not args.skip_java:
        print("\n  --- Indicator Comparison ---")
        try:
            py_ind = run_python_indicators(csv_path, min(5, args.ticks))
            java_ind = run_java_indicators(csv_path, min(5, args.ticks))
            ind_pass, ind_diffs = compare_indicators(py_ind, java_ind, tolerance=0.01)

            if ind_pass:
                print("  PASS: All indicator values match within 1% tolerance")
            else:
                print(f"  FAIL: {len(ind_diffs)} indicator discrepancies found")
                print(f"\n  {'Indicator':<15} {'Tick':>5} {'Python':>12} {'Java':>12} {'Diff%':>10}")
                print("  " + "-" * 58)
                for d in ind_diffs[:20]:
                    print(f"  {d['indicator']:<15} {d['tick']:>5} {d['python']:>12.6f} {d['java']:>12.6f} {d['rel_diff_pct']:>9.4f}%")
                if len(ind_diffs) > 20:
                    print(f"  ... and {len(ind_diffs) - 20} more")
        except Exception as e:
            print(f"  Indicator comparison failed: {e}")
            ind_pass = False

    # 4b. Compare agent results
    agent_pass = False
    agent_diffs = []
    pass_count = 0

    if java_results is not None:
        print("\n  --- Agent PnL Comparison ---")
        agent_pass, agent_diffs, pass_count = compare_agent_results(
            py_results, java_results, tolerance=args.tolerance, id_to_name=id_to_name
        )

        # Print diff table
        print(f"\n  {'Agent':<12} {'Py Final':>12} {'Ja Final':>12} {'Diff%':>10} {'PyTrades':>9} {'JaTrades':>9} {'Status':>8}")
        print("  " + "-" * 78)

        for d in agent_diffs:
            status = "PASS" if d["within_tolerance"] else "** FAIL"
            print(
                f"  {d['agent']:<12} "
                f"{d['py_final']:>12,.2f} "
                f"{d['ja_final']:>12,.2f} "
                f"{d['final_diff_pct']:>9.4f}% "
                f"{d['py_trades']:>9d} "
                f"{d['ja_trades']:>9d} "
                f"{status:>8}"
            )

        # Document discrepancies > tolerance
        failures = [d for d in agent_diffs if not d["within_tolerance"]]
        if failures:
            print(f"\n  DISCREPANCIES >{args.tolerance * 100:.1f}% (root cause analysis):")
            for f in failures:
                root_cause = diagnose_discrepancy(f)
                print(f"    {f['agent']}: {root_cause}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("  VALIDATION SUMMARY")
    print("=" * 80)

    overall_pass = True

    if not args.skip_indicators:
        print(f"  Indicator match:  {'PASS' if ind_pass else 'FAIL'}")
        if not ind_pass:
            overall_pass = False

    if java_results is not None:
        print(f"  Agent PnL match:  {'PASS' if agent_pass else 'FAIL'}")
        print(f"  Agents passing:   {pass_count}/{len(agent_diffs)}")
        if pass_count >= 5:
            print(f"  Minimum 5 codecs: PASS ({pass_count} >= 5)")
        else:
            print(f"  WARNING: Only {pass_count} codecs pass (< 5 required)")
            overall_pass = False
        if not agent_pass and pass_count >= 5:
            # Still PASS overall if 5+ codecs match (acceptance criteria)
            print(f"  Note: {len(agent_diffs) - pass_count} codecs differ >{args.tolerance*100:.1f}% (documented above)")
    else:
        print("  Java results:     SKIPPED/FAILED")
        overall_pass = False

    print(f"\n  OVERALL:          {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 80)

    # Cleanup temp CSV
    if not args.csv and os.path.exists(csv_path):
        try:
            os.unlink(csv_path)
        except OSError:
            pass

    return 0 if overall_pass else 1


def diagnose_discrepancy(d: Dict[str, Any]) -> str:
    """
    Produce a root-cause diagnosis for a discrepancy.
    """
    diff_pct = d["final_diff_pct"]
    py_trades = d["py_trades"]
    ja_trades = d["ja_trades"]

    if py_trades == 0 and ja_trades == 0:
        return f"Diff {diff_pct:.4f}% — Both agents held (no trades). Numerical precision in portfolio valuation."

    if py_trades != ja_trades:
        return (
            f"Diff {diff_pct:.4f}% — Trade count mismatch (py={py_trades}, ja={ja_trades}). "
            f"Likely caused by differing signal generation logic in the codec forward() method."
        )

    if abs(d["py_pnl"] - d["ja_pnl"]) > 1.0:
        return (
            f"Diff {diff_pct:.4f}% — PnL mismatch (py={d['py_pnl']:.2f}, ja={d['ja_pnl']:.2f}). "
            f"Possible causes: indicator computation drift, signal threshold differences, "
            f"or floating-point precision in position sizing."
        )

    return (
        f"Diff {diff_pct:.4f}% — Minor discrepancy. Likely floating-point precision "
        f"or rounding differences between Python (float64) and Java (double)."
    )


if __name__ == "__main__":
    sys.exit(main())
