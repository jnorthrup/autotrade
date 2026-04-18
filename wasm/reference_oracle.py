#!/usr/bin/env python3
"""
Deterministic Reference Specification (Oracle) for Dreamer WASM Kernel
Version: 1.0.0
Precision: binary64 (IEEE 754 double-precision)

This module provides reference implementations for all promoted calculations
in the WASM kernel. It serves as the oracle for:
- Kernel verification against known-good binary64 results
- Serialization validation
- End-to-end integration tests

The reference implementations use Python's float (binary64) and are designed
to exactly match the f64 semantics of the Rust/WASM kernel.

NaN and Signed-Zero Rules:
- Operations producing NaN: 0.0/0.0, inf/inf, inf - inf, 0.0 * inf
- NaN propagates through all operations (NaN op x = NaN)
- Signed zero: -0.0 is distinct from +0.0 in comparisons and sign bit
- Division by zero: x/0.0 = inf (with sign), 0.0/0.0 = NaN
"""

import math
import struct
from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass
from enum import IntEnum


# ============================================================================
# IEEE 754 Binary64 Constants and Utilities
# ============================================================================

class IEEE754:
    """IEEE 754 binary64 (double-precision) constants and utilities."""
    
    # Bit patterns for special values
    POSITIVE_ZERO = 0.0
    NEGATIVE_ZERO = -0.0
    POSITIVE_INFINITY = float('inf')
    NEGATIVE_INFINITY = float('-inf')
    QUIET_NAN = float('nan')
    
    # Smallest positive normal number
    MIN_POSITIVE_NORMAL = 2.0**-1022
    
    # Smallest positive subnormal number
    MIN_POSITIVE_SUBNORMAL = 2.0**-1074
    
    # Largest finite number
    MAX_FINITE = (2.0 - 2.0**-52) * 2.0**1023
    
    # Machine epsilon (unit roundoff)
    EPSILON = 2.0**-52
    
    @staticmethod
    def is_signed_zero(x: float) -> bool:
        """Check if x is -0.0 (negative zero)."""
        return x == 0.0 and struct.unpack('>Q', struct.pack('>d', x))[0] >> 63 == 1
    
    @staticmethod
    def is_positive_zero(x: float) -> bool:
        """Check if x is +0.0 (positive zero)."""
        return x == 0.0 and struct.unpack('>Q', struct.pack('>d', x))[0] >> 63 == 0
    
    @staticmethod
    def is_subnormal(x: float) -> bool:
        """Check if x is a subnormal (denormal) number."""
        if not math.isfinite(x) or x == 0.0:
            return False
        abs_x = abs(x)
        return 0.0 < abs_x < IEEE754.MIN_POSITIVE_NORMAL
    
    @staticmethod
    def get_sign_bit(x: float) -> int:
        """Get the sign bit of x (0 for positive, 1 for negative)."""
        return struct.unpack('>Q', struct.pack('>d', x))[0] >> 63
    
    @staticmethod
    def same_bits(a: float, b: float) -> bool:
        """Check if two floats have identical bit representations."""
        return struct.pack('>d', a) == struct.pack('>d', b)


# ============================================================================
# Regime Encoding (must match dreamer_kernel.rs)
# ============================================================================

class Regime(IntEnum):
    """Market regime classification encoding."""
    UNKNOWN = 0
    CRAB_CHOP = 1
    BULL_RUSH = 2
    BEAR_CRASH = 3
    STEADY_GROWTH = 4
    VOLATILE_CHOP = 5


# ============================================================================
# Core Calculation Results
# ============================================================================

@dataclass(frozen=True)
class WindowFeatures:
    """Result of rolling window feature computation."""
    mean_price: float
    price_variance: float
    vwap: float
    latest_price: float
    price_momentum: float
    mean_volume: float


@dataclass(frozen=True)
class PortfolioFeatures:
    """Result of portfolio feature computation."""
    deviation_percent: float
    crash_active: float  # 0.0 or 1.0
    declining_count: float
    managed_baseline: float
    baseline_diff: float
    deviations: Tuple[float, ...]
    harvest_candidates: Tuple[float, ...]  # 0.0 or 1.0 flags
    rebalance_candidates: Tuple[float, ...]  # 0.0 or 1.0 flags


@dataclass(frozen=True)
class DefectScanResult:
    """Result of shadow defect scan."""
    is_defective: float  # 0.0 or 1.0
    max_drawdown: float
    trigger_hits: float


@dataclass(frozen=True)
class RegimeResult:
    """Result of regime computation."""
    regime: float  # Encoded as Regime enum
    roi: float
    volatility: float
    mean: float


# ============================================================================
# Reference Implementation: Rolling Window Features
# ============================================================================

def compute_window_features_reference(
    prices: List[float],
    volumes: List[float],
    window_size: int,
) -> WindowFeatures:
    """
    Reference implementation of rolling window feature computation.
    
    Matches the Rust/WASM kernel semantic exactly:
    - Uses f64 for all intermediate results
    - Handles empty windows (returns NaN for all features)
    - Variance computation: E[x²] - E[x]² with negative variance clamping
    - VWAP: notional / volume_sum (NaN if volume_sum == 0)
    """
    n = min(len(prices), len(volumes), window_size)
    
    if n == 0:
        nan = math.nan
        return WindowFeatures(nan, nan, nan, nan, nan, nan)
    
    # Use last n elements
    window_prices = prices[-n:]
    window_volumes = volumes[-n:]
    
    # Accumulate statistics
    nf = float(n)
    sum_price = 0.0
    sum_volume = 0.0
    notional = 0.0
    
    for price, volume in zip(window_prices, window_volumes):
        sum_price += price
        sum_volume += volume
        notional += price * volume
    
    # Compute variance using Welford's algorithm for numerical stability
    mean = 0.0
    m2 = 0.0
    for i, price in enumerate(window_prices):
        delta = price - mean
        mean += delta / (i + 1)
        delta2 = price - mean
        m2 += delta * delta2
    
    variance = m2 / nf if n > 1 else 0.0
    
    # Compute other features at f64 precision
    mean_price = sum_price / nf
    vwap = notional / sum_volume if sum_volume != 0.0 else math.nan
    oldest_price = window_prices[0]
    latest_price = window_prices[-1]
    price_momentum = latest_price - oldest_price
    mean_volume = sum_volume / nf
    
    return WindowFeatures(
        mean_price=mean_price,
        price_variance=variance,
        vwap=vwap,
        latest_price=latest_price,
        price_momentum=price_momentum,
        mean_volume=mean_volume,
    )


# ============================================================================
# Reference Implementation: Portfolio Features
# ============================================================================

def compute_portfolio_features_reference(
    values: List[float],
    baselines: List[float],
    cash_balance: float,
    harvest_trigger: float,
    rebalance_trigger: float,
    cp_trigger_asset_percent: float,
    cp_trigger_min_negative_dev: float,
) -> PortfolioFeatures:
    """
    Reference implementation of portfolio feature computation.
    
    Matches the Rust/WASM kernel semantics exactly:
    - Deviation per asset: (value - baseline) / baseline (0.0 if baseline <= 0)
    - Harvest candidate: deviation >= harvest_trigger
    - Rebalance candidate: deviation <= -rebalance_trigger
    - Crash protection: percent_declining >= cp_trigger_asset_percent
    """
    asset_count = len(values)
    if len(baselines) != asset_count:
        raise ValueError("values and baselines must have same length")
    
    deviations = []
    harvest_candidates = []
    rebalance_candidates = []
    
    total_baseline_diff = 0.0
    total_managed_baseline = 0.0
    declining_count = 0.0
    
    for value, baseline in zip(values, baselines):
        # Compute deviation
        if baseline > 0.0:
            deviation = (value - baseline) / baseline
        else:
            deviation = 0.0
        
        deviations.append(deviation)
        
        # Harvest candidate
        if deviation >= harvest_trigger:
            harvest_candidates.append(1.0)
        else:
            harvest_candidates.append(0.0)
        
        # Rebalance candidate
        if deviation <= -rebalance_trigger:
            rebalance_candidates.append(1.0)
        else:
            rebalance_candidates.append(0.0)
        
        # Crash protection counting
        if baseline > 0.0:
            total_baseline_diff += (value - baseline)
            total_managed_baseline += baseline
            
            if deviation <= cp_trigger_min_negative_dev:
                declining_count += 1.0
    
    # Compute aggregate metrics
    if total_managed_baseline > 0.0:
        deviation_percent = (total_baseline_diff / total_managed_baseline) * 100.0
    else:
        deviation_percent = 0.0
    
    n = float(asset_count)
    if total_managed_baseline > 0.0:
        percent_declining = declining_count / n
        crash_active = 1.0 if percent_declining >= cp_trigger_asset_percent else 0.0
    else:
        crash_active = 0.0
    
    return PortfolioFeatures(
        deviation_percent=deviation_percent,
        crash_active=crash_active,
        declining_count=declining_count,
        managed_baseline=total_managed_baseline,
        baseline_diff=total_baseline_diff,
        deviations=tuple(deviations),
        harvest_candidates=tuple(harvest_candidates),
        rebalance_candidates=tuple(rebalance_candidates),
    )


# ============================================================================
# Reference Implementation: Defect Scan
# ============================================================================

def scan_for_defects_reference(
    prices: List[float],
    rebalance_trigger: float,
    crash_threshold: float,
) -> DefectScanResult:
    """
    Reference implementation of shadow defect scan.
    
    Matches the Rust/WASM kernel semantics exactly:
    - Tracks max_price seen so far
    - Trigger: (current - max) / max < rebalance_trigger (negative trigger)
    - On trigger, look ahead for drops > crash_threshold
    - Returns: is_defective (0.0 or 1.0), max_drawdown, trigger_hits
    """
    price_count = len(prices)
    
    if price_count < 2:
        return DefectScanResult(0.0, 0.0, 0.0)
    
    max_price = prices[0]
    is_defective = 0.0
    max_drawdown = 0.0
    trigger_hits = 0.0
    
    for i in range(price_count):
        current_price = prices[i]
        
        if current_price > max_price:
            max_price = current_price
        
        # Check if trigger condition is met
        if max_price != 0.0:
            deviation = (current_price - max_price) / max_price
        else:
            deviation = math.nan
        
        if deviation < rebalance_trigger:
            trigger_hits += 1.0
            
            # Look ahead for crash
            min_future_price = current_price
            for j in range(i + 1, price_count):
                if prices[j] < min_future_price:
                    min_future_price = prices[j]
            
            if current_price != 0.0:
                subsequent_drop = (min_future_price - current_price) / current_price
            else:
                subsequent_drop = math.nan
            
            if subsequent_drop < -crash_threshold:
                is_defective = 1.0
                drawdown = -subsequent_drop
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
    
    return DefectScanResult(is_defective, max_drawdown, trigger_hits)


# ============================================================================
# Reference Implementation: Regime Detection
# ============================================================================

def compute_regime_reference(
    history: List[float],
    current_price: float,
    start_price: float,
) -> RegimeResult:
    """
    Reference implementation of regime detection.
    
    Matches the Rust/WASM kernel semantics exactly:
    - Requires at least 50 history points and positive start_price
    - ROI = (current - start) / start
    - Volatility = sqrt(variance) / mean
    - Regime classification uses f64 thresholds
    """
    history_len = len(history)
    
    if history_len < 50 or start_price <= 0.0:
        return RegimeResult(float(Regime.UNKNOWN), 0.0, 0.0, 0.0)
    
    roi = (current_price - start_price) / start_price
    
    # Compute mean
    total = 0.0
    for price in history:
        total += price
    mean = total / float(history_len)
    
    # Compute variance
    sum_sq_diff = 0.0
    for price in history:
        diff = price - mean
        sum_sq_diff += diff * diff
    variance = sum_sq_diff / float(history_len)
    
    if mean > 0.0:
        volatility = math.sqrt(variance) / mean
    else:
        volatility = 0.0
    
    # Classify regime using same thresholds as kernel
    if roi > 0.05 and volatility > 0.02:
        regime = Regime.BULL_RUSH
    elif roi < -0.05 and volatility > 0.02:
        regime = Regime.BEAR_CRASH
    elif roi > 0.02 and volatility < 0.01:
        regime = Regime.STEADY_GROWTH
    elif volatility > 0.05:
        regime = Regime.VOLATILE_CHOP
    else:
        regime = Regime.CRAB_CHOP
    
    return RegimeResult(
        regime=float(regime),
        roi=roi,
        volatility=volatility,
        mean=mean,
    )


# ============================================================================
# Test Corpus: Normal Market Paths
# ============================================================================

NORMAL_MARKET_PATHS = {
    "simple_uptrend": {
        "description": "Simple uptrend with minimal volatility",
        "prices": [100.0 + i * 0.5 for i in range(100)],
        "volumes": [1000.0 + i * 10 for i in range(100)],
        "window_size": 20,
    },
    "simple_downtrend": {
        "description": "Simple downtrend with minimal volatility",
        "prices": [150.0 - i * 0.3 for i in range(100)],
        "volumes": [5000.0 - i * 20 for i in range(100)],
        "window_size": 20,
    },
    "sideways_market": {
        "description": "Sideways market with oscillation",
        "prices": [100.0 + 5.0 * math.sin(i * 0.2) for i in range(100)],
        "volumes": [2000.0] * 100,
        "window_size": 20,
    },
    "volatile_market": {
        "description": "High volatility market",
        "prices": [100.0 + 15.0 * math.sin(i * 0.5) + 5.0 * (i % 3 - 1) for i in range(100)],
        "volumes": [1000.0 + 500.0 * (i % 5) for i in range(100)],
        "window_size": 20,
    },
    "low_liquidity": {
        "description": "Very low volume scenario",
        "prices": [100.0 + i * 0.1 for i in range(50)],
        "volumes": [0.001] * 50,
        "window_size": 10,
    },
    "high_precision_prices": {
        "description": "Prices with many decimal places",
        "prices": [123.456789012345 + i * 0.0000001 for i in range(100)],
        "volumes": [1000.0] * 100,
        "window_size": 20,
    },
}


# ============================================================================
# Test Corpus: 2-Degree-of-Contact Boundary Scenarios
# ============================================================================

CONTACT_2_BOUNDARIES = {
    "window_exactly_full": {
        "description": "Window size equals data length (exactly at capacity)",
        "prices": [100.0 + i * 1.0 for i in range(20)],
        "volumes": [1000.0] * 20,
        "window_size": 20,
    },
    "window_one_under": {
        "description": "Window size one less than data length",
        "prices": [100.0 + i * 1.0 for i in range(21)],
        "volumes": [1000.0] * 21,
        "window_size": 20,
    },
    "window_larger_than_data": {
        "description": "Window size larger than available data",
        "prices": [100.0 + i * 1.0 for i in range(10)],
        "volumes": [1000.0] * 10,
        "window_size": 20,
    },
    "empty_window": {
        "description": "Empty data (boundary case)",
        "prices": [],
        "volumes": [],
        "window_size": 10,
    },
    "single_point": {
        "description": "Single data point",
        "prices": [100.0],
        "volumes": [1000.0],
        "window_size": 10,
    },
    "trigger_exactly_at_threshold": {
        "description": "Deviation exactly at harvest trigger threshold",
        "prices": [100.0, 90.0, 80.0],
        "rebalance_trigger": -0.1,  # Exactly -10%
        "crash_threshold": 0.01,
    },
}


# ============================================================================
# Test Corpus: IEEE 754 Corner Cases
# ============================================================================

IEEE_754_CORNER_CASES = {
    "positive_infinity_price": {
        "description": "Positive infinity in price data",
        "prices": [100.0, float('inf'), 102.0],
        "volumes": [1000.0, 1000.0, 1000.0],
        "window_size": 3,
    },
    "negative_infinity_price": {
        "description": "Negative infinity in price data",
        "prices": [100.0, float('-inf'), 102.0],
        "volumes": [1000.0, 1000.0, 1000.0],
        "window_size": 3,
    },
    "nan_in_prices": {
        "description": "NaN in price data",
        "prices": [100.0, float('nan'), 102.0],
        "volumes": [1000.0, 1000.0, 1000.0],
        "window_size": 3,
    },
    "nan_in_volumes": {
        "description": "NaN in volume data",
        "prices": [100.0, 101.0, 102.0],
        "volumes": [1000.0, float('nan'), 1000.0],
        "window_size": 3,
    },
    "zero_volume": {
        "description": "Zero volume (VWAP boundary)",
        "prices": [100.0, 101.0, 102.0],
        "volumes": [0.0, 0.0, 0.0],
        "window_size": 3,
    },
    "negative_zero": {
        "description": "Negative zero values",
        "prices": [100.0, -0.0, 102.0],
        "volumes": [1000.0, 1000.0, 1000.0],
        "window_size": 3,
    },
    "subnormal_prices": {
        "description": "Subnormal (denormal) prices",
        "prices": [IEEE754.MIN_POSITIVE_SUBNORMAL, IEEE754.MIN_POSITIVE_SUBNORMAL * 2, 
                   IEEE754.MIN_POSITIVE_SUBNORMAL * 3],
        "volumes": [1000.0, 1000.0, 1000.0],
        "window_size": 3,
    },
    "max_finite_value": {
        "description": "Maximum finite f64 values",
        "prices": [IEEE754.MAX_FINITE, IEEE754.MAX_FINITE, IEEE754.MAX_FINITE],
        "volumes": [1000.0, 1000.0, 1000.0],
        "window_size": 3,
    },
    "mixed_inf_nan": {
        "description": "Mix of inf and NaN",
        "prices": [float('inf'), float('nan'), float('-inf')],
        "volumes": [1000.0, 1000.0, 1000.0],
        "window_size": 3,
    },
    "signed_zero_arithmetic": {
        "description": "Test signed zero behavior",
        "values": [100.0, 100.0],
        "baselines": [-0.0, 0.0],  # One positive, one negative zero
        "cash_balance": 0.0,
        "harvest_trigger": 0.0,
        "rebalance_trigger": 0.0,
        "cp_trigger_asset_percent": 0.5,
        "cp_trigger_min_negative_dev": -0.1,
    },
    "division_by_zero": {
        "description": "Division by zero scenarios",
        "values": [100.0, 200.0],
        "baselines": [0.0, 0.0],  # Will cause division by zero
        "cash_balance": 0.0,
        "harvest_trigger": 0.0,
        "rebalance_trigger": 0.0,
        "cp_trigger_asset_percent": 0.5,
        "cp_trigger_min_negative_dev": -0.1,
    },
}


# ============================================================================
# Expected Results Database
# ============================================================================

def generate_expected_results() -> dict:
    """
    Generate expected results for all test corpus entries.
    These are the canonical binary64 reference values.
    """
    results = {
        "window_features": {},
        "portfolio_features": {},
        "defect_scan": {},
        "regime": {},
    }
    
    # Normal market paths - window features
    for name, data in NORMAL_MARKET_PATHS.items():
        if "prices" in data and "volumes" in data and "window_size" in data:
            result = compute_window_features_reference(
                data["prices"],
                data["volumes"],
                data["window_size"],
            )
            results["window_features"][name] = result
    
    # 2-degree-of-contact boundaries - window features
    for name, data in CONTACT_2_BOUNDARIES.items():
        if "prices" in data and "volumes" in data and "window_size" in data:
            result = compute_window_features_reference(
                data["prices"],
                data["volumes"],
                data["window_size"],
            )
            results["window_features"][name] = result
    
    # IEEE 754 corner cases - window features
    for name, data in IEEE_754_CORNER_CASES.items():
        if "prices" in data and "volumes" in data and "window_size" in data:
            result = compute_window_features_reference(
                data["prices"],
                data["volumes"],
                data["window_size"],
            )
            results["window_features"][name] = result
    
    return results


# ============================================================================
# Main Entry Point: Reference Oracle
# ============================================================================

class ReferenceOracle:
    """
    Deterministic reference oracle for WASM kernel verification.
    
    This class provides the single source of truth for expected binary64
    calculation results. It is reusable across:
    - Kernel unit tests
    - Serialization validation
    - End-to-end integration tests
    """
    
    def __init__(self):
        self.expected_results = generate_expected_results()
    
    def get_expected_window_features(self, test_case: str) -> Optional[WindowFeatures]:
        """Get expected window features for a named test case."""
        return self.expected_results["window_features"].get(test_case)
    
    def verify_window_features(
        self,
        test_case: str,
        actual: WindowFeatures,
        tolerance: float = 1e-15,
    ) -> Tuple[bool, List[str]]:
        """
        Verify actual window features against expected values.
        
        Returns (pass, [failures])
        NaN values are compared using math.isnan on both sides.
        """
        expected = self.get_expected_window_features(test_case)
        if expected is None:
            return False, [f"No expected results for test case: {test_case}"]
        
        failures = []
        fields = ['mean_price', 'price_variance', 'vwap', 'latest_price', 
                  'price_momentum', 'mean_volume']
        
        for field in fields:
            exp_val = getattr(expected, field)
            act_val = getattr(actual, field)
            
            # Handle NaN comparison
            if math.isnan(exp_val) and math.isnan(act_val):
                continue
            
            # Handle infinite values
            if math.isinf(exp_val) and math.isinf(act_val):
                if (exp_val > 0) == (act_val > 0):  # Same sign
                    continue
            
            # Finite comparison
            if not math.isclose(exp_val, act_val, rel_tol=tolerance, abs_tol=tolerance):
                failures.append(
                    f"{field}: expected {exp_val}, got {act_val}"
                )
        
        return len(failures) == 0, failures
    
    def compute_reference(
        self,
        kernel_type: str,
        inputs: dict,
    ):
        """Compute reference result for given kernel type and inputs."""
        if kernel_type == "window_features":
            return compute_window_features_reference(
                inputs["prices"],
                inputs["volumes"],
                inputs.get("window_size", 20),
            )
        elif kernel_type == "portfolio_features":
            return compute_portfolio_features_reference(
                inputs["values"],
                inputs["baselines"],
                inputs.get("cash_balance", 0.0),
                inputs.get("harvest_trigger", 0.0),
                inputs.get("rebalance_trigger", 0.0),
                inputs.get("cp_trigger_asset_percent", 0.5),
                inputs.get("cp_trigger_min_negative_dev", -0.1),
            )
        elif kernel_type == "defect_scan":
            return scan_for_defects_reference(
                inputs["prices"],
                inputs["rebalance_trigger"],
                inputs["crash_threshold"],
            )
        elif kernel_type == "regime":
            return compute_regime_reference(
                inputs["history"],
                inputs["current_price"],
                inputs["start_price"],
            )
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")


# ============================================================================
# Serialization for cross-language compatibility
# ============================================================================

def serialize_reference_result(result) -> dict:
    """
    Serialize a reference result to a JSON-compatible dict with exact
    binary64 hex representation for precise cross-language comparison.
    """
    def float_to_hex(f: float) -> str:
        """Convert float to IEEE 754 hex representation."""
        return format(struct.unpack('>Q', struct.pack('>d', f))[0], '016x')
    
    def serialize_value(v):
        if isinstance(v, float):
            return {
                "value": v,
                "hex": float_to_hex(v),
                "is_nan": math.isnan(v),
                "is_inf": math.isinf(v),
                "is_neg_zero": IEEE754.is_signed_zero(v),
            }
        elif isinstance(v, tuple):
            return [serialize_value(x) for x in v]
        else:
            return v
    
    if isinstance(result, (WindowFeatures, PortfolioFeatures, 
                           DefectScanResult, RegimeResult)):
        return {
            "type": result.__class__.__name__,
            "fields": {
                k: serialize_value(v) for k, v in result.__dict__.items()
            }
        }
    return serialize_value(result)


def deserialize_reference_result(data: dict):
    """Deserialize reference result from JSON-compatible dict."""
    def parse_value(v):
        if isinstance(v, dict) and "hex" in v:
            return struct.unpack('>d', bytes.fromhex(v["hex"]))[0]
        elif isinstance(v, list):
            return tuple(parse_value(x) for x in v)
        return v
    
    if "fields" in data:
        fields = {k: parse_value(v) for k, v in data["fields"].items()}
        type_name = data["type"]
        if type_name == "WindowFeatures":
            return WindowFeatures(**fields)
        elif type_name == "PortfolioFeatures":
            return PortfolioFeatures(**fields)
        elif type_name == "DefectScanResult":
            return DefectScanResult(**fields)
        elif type_name == "RegimeResult":
            return RegimeResult(**fields)
    return data


# ============================================================================
# Module Entry Point
# ============================================================================

if __name__ == "__main__":
    # Generate and print expected results for all test cases
    oracle = ReferenceOracle()
    
    print("=" * 70)
    print("DREAMER WASM KERNEL - REFERENCE ORACLE RESULTS")
    print("Version: 1.0.0")
    print("Precision: binary64 (IEEE 754 double-precision)")
    print("=" * 70)
    
    print("\n--- Normal Market Paths ---")
    for name in NORMAL_MARKET_PATHS:
        result = oracle.get_expected_window_features(name)
        if result:
            print(f"\n{name}:")
            print(f"  mean_price: {result.mean_price}")
            print(f"  price_variance: {result.price_variance}")
            print(f"  vwap: {result.vwap}")
    
    print("\n--- 2-Degree-of-Contact Boundaries ---")
    for name in CONTACT_2_BOUNDARIES:
        result = oracle.get_expected_window_features(name)
        if result:
            print(f"\n{name}:")
            print(f"  mean_price: {result.mean_price}")
            print(f"  vwap: {result.vwap}")
    
    print("\n--- IEEE 754 Corner Cases ---")
    for name in IEEE_754_CORNER_CASES:
        result = oracle.get_expected_window_features(name)
        if result:
            print(f"\n{name}:")
            print(f"  mean_price: {result.mean_price}")
            print(f"  vwap: {result.vwap}")
    
    print("\n" + "=" * 70)
    print("Reference oracle ready for kernel verification")
    print("=" * 70)
