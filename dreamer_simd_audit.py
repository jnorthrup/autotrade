from __future__ import annotations

from dataclasses import dataclass
from textwrap import indent
from typing import Tuple


@dataclass(frozen=True)
class KernelCandidate:
    rank: int
    name: str
    source: str
    cpu_cost_score: int
    reuse_score: int
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    call_frequency: str
    notes: str


@dataclass(frozen=True)
class OutOfScopeArea:
    category: str
    reason: str
    examples: Tuple[str, ...]


@dataclass(frozen=True)
class SimdAudit:
    title: str
    scope_statement: str
    in_scope_kernels: Tuple[KernelCandidate, ...]
    out_of_scope_areas: Tuple[OutOfScopeArea, ...]
    boundary_notes: Tuple[str, ...]


AUDIT = SimdAudit(
    title="Dreamer 1.999 WASM SIMD audit",
    scope_statement=(
        "The SIMD boundary is the reusable numeric kernel layer only. "
        "Host orchestration, IPC, file I/O, state machines, logging, and market access "
        "stay on the JS/Python side and are explicitly out of scope."
    ),
    in_scope_kernels=(
        KernelCandidate(
            rank=1,
            name="showdown.indicators feature kernel family",
            source="showdown/indicators.py: IndicatorComputer._compute_indicators + _rsi + _atr + _stochastic + _adx + _rolling_ema_series + _vwap + _momentum",
            cpu_cost_score=5,
            reuse_score=5,
            inputs=(
                "per-pair rolling OHLCV buffers (opens/highs/lows/closes/volumes/typicals)",
                "latest price and volume tick",
            ),
            outputs=(
                "market_data dict with SMA/EMA/MACD/RSI/Bollinger/ATR/Stochastic/ADX/VWAP/momentum/log_return",
            ),
            call_frequency=(
                "once per tick per pair per agent; reused by every codec forward pass"
            ),
            notes=(
                "This is the clearest WASM auto-vectorization target in the repo: multiple "
                "dense numeric passes over contiguous arrays, no orchestration dependency. "
                "The wrapper compute_market_data() and the agent-level packing logic are not "
                "the SIMD target; the indicator math underneath them is."
            ),
        ),
        KernelCandidate(
            rank=2,
            name="Dreamer portfolio feature and risk aggregation slice",
            source="dreamer/Dreamer 1.2.js: TradingEngine.update numeric reductions and per-row score math",
            cpu_cost_score=4,
            reuse_score=5,
            inputs=(
                "portfolioSummary rows (Symbol, Price, Value, Baseline, usdValueNum)",
                "cashBalance and holdings-derived state",
                "genome thresholds (harvest/rebalance/crash-protection params)",
                "lastCyclePrices and baseline maps",
            ),
            outputs=(
                "currentPortfolioDeviationPercent",
                "crash-protection counts/flags",
                "harvest/rebalance candidate scores and trigger checks",
            ),
            call_frequency=(
                "once per live/shadow update cycle, and again for every simulated tick "
                "inside ScientificOptimizer.evaluateGenome()"
            ),
            notes=(
                "Keep the state machine, trade placement, logging, and persistence out of the "
                "kernel. The SIMD candidate is the repeated numeric sweep over the asset rows "
                "and the reduction math that feeds the risk gates."
            ),
        ),
        KernelCandidate(
            rank=3,
            name="Dreamer shadow defect scan",
            source="dreamer/Dreamer 1.2.js: TradingEngine.isDefective",
            cpu_cost_score=4,
            reuse_score=4,
            inputs=(
                "priceHistoryBuffer for the assigned asset",
                "rebalance trigger threshold from genome/overrides",
            ),
            outputs=("boolean defective flag",),
            call_frequency=(
                "every shadow update cycle and every shadow lifecycle validation pass"
            ),
            notes=(
                "This is a pure rolling-window scan with nested look-ahead. It is a good SIMD "
                "candidate once the history is moved into contiguous numeric buffers instead of "
                "object ticks. The shadow life-cycle decisions themselves remain host logic."
            ),
        ),
        KernelCandidate(
            rank=4,
            name="Dreamer regime statistics kernel",
            source="dreamer/Dreamer 1.2.js: RegimeDetector.analyze",
            cpu_cost_score=3,
            reuse_score=4,
            inputs=("per-symbol history array of close prices",),
            outputs=("regime label (CRAB_CHOP / BULL_RUSH / BEAR_CRASH / STEADY_GROWTH / VOLATILE_CHOP / UNKNOWN)",),
            call_frequency=(
                "per symbol whenever regime analysis is refreshed; intended to run on every "
                "portfolio cycle for active symbols"
            ),
            notes=(
                "This is numeric but smaller than the indicator family and the portfolio sweep. "
                "It belongs in the SIMD lane only if regime analysis is batched across many assets."
            ),
        ),
    ),
    out_of_scope_areas=(
        OutOfScopeArea(
            category="orchestration",
            reason=(
                "These functions decide when to call the kernels, but they do not perform "
                "the reusable numeric work themselves."
            ),
            examples=(
                "showdown/agent.py: Agent.step / Agent.reset_runtime_state",
                "dreamer/Dreamer 1.2.js: mainLoop()",
                "dreamer/Dreamer 1.2.js: ScientificOptimizer.run()",
                "dreamer/Dreamer 1.2.js: LegionManager.heartbeat(), deploySwarm(), marchLegion()",
                "dreamer/Dreamer 1.2.js: getNextCombinatorialCandidate()",
            ),
        ),
        OutOfScopeArea(
            category="ipc",
            reason=(
                "Message passing and worker control are control-plane work, not SIMD math."
            ),
            examples=(
                "dreamer/Dreamer 1.2.js: process.on('message', ...)",
                "dreamer/Dreamer 1.2.js: process.send({...}) in the optimizer worker",
                "dreamer/Dreamer 1.2.js: dispatchToDreamer(msg)",
            ),
        ),
        OutOfScopeArea(
            category="file_io",
            reason=(
                "Disk and network access should remain on the host side so the kernel stays "
                "pure, batchable, and deterministic."
            ),
            examples=(
                "dreamer/Dreamer 1.2.js: loadRecentMarketData(), appendMarketData()",
                "dreamer/Dreamer 1.2.js: loadState(), saveState(), saveEngineState()",
                "dreamer/Dreamer 1.2.js: saveBrainScan()",
                "dreamer/Dreamer 1.2.js: RobinhoodAPI request methods and quote/holding fetches",
                "showdown/indicators.py: compute_market_data() singleton wrapper if treated as an I/O-facing host API",
            ),
        ),
        OutOfScopeArea(
            category="state_machine",
            reason=(
                "Mode selection, promotion, lifecycle transitions, and recovery logic are "
                "policy decisions; the SIMD kernel should only receive the numeric state it needs."
            ),
            examples=(
                "dreamer/Dreamer 1.2.js: ScientificOptimizer.evaluateGenome()",
                "dreamer/Dreamer 1.2.js: TradingEngine.update() orchestration branches",
                "dreamer/Dreamer 1.2.js: portfolioHarvestState / trailingState / rebalanceState transitions",
                "showdown/agent.py: portfolio bookkeeping and trade execution",
            ),
        ),
    ),
    boundary_notes=(
        "The WASM boundary should accept contiguous Float64 arrays or array views, not object rows.",
        "Kernel outputs should be numeric buffers or tightly packed structs; host code can rehydrate dicts or rows later.",
        "The wrapper functions are allowed to route data, allocate buffers, and persist state, but they are not part of the SIMD kernel.",
    ),
)


def _format_list(items: Tuple[str, ...], indent_level: int = 4) -> str:
    prefix = " " * indent_level + "- "
    return "\n".join(f"{prefix}{item}" for item in items)


def render_markdown(audit: SimdAudit = AUDIT) -> str:
    lines = [
        f"# {audit.title}",
        "",
        "## Scope",
        audit.scope_statement,
        "",
        "## Ranked SIMD candidates",
    ]

    for kernel in sorted(audit.in_scope_kernels, key=lambda k: k.rank):
        lines.extend(
            [
                "",
                f"### Rank {kernel.rank}: {kernel.name}",
                f"Source: {kernel.source}",
                f"CPU cost: {kernel.cpu_cost_score}/5",
                f"Reuse frequency: {kernel.reuse_score}/5",
                "Inputs:",
                _format_list(kernel.inputs),
                "Outputs:",
                _format_list(kernel.outputs),
                f"Call frequency: {kernel.call_frequency}",
                "Notes:",
                indent(kernel.notes, "    "),
            ]
        )

    lines.extend([
        "",
        "## Explicit out-of-SIMD scope",
    ])

    for area in audit.out_of_scope_areas:
        lines.extend(
            [
                "",
                f"### {area.category}",
                area.reason,
                "Examples:",
                _format_list(area.examples),
            ]
        )

    lines.extend(
        [
            "",
            "## Boundary notes",
        ]
    )
    for note in audit.boundary_notes:
        lines.append(f"- {note}")

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    print(render_markdown())
