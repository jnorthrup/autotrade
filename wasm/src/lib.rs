//! Dreamer WASM Kernel - SIMD-Optimized Implementation (Stable Rust)
//! 
//! This module implements the exact numeric hot paths promoted into WASM
//! with full auto-vectorization support for WASM SIMD f64 operations.
//! 
//! Build: cargo build --target wasm32-unknown-unknown --release
//! SIMD:  -C target-feature=+simd128
//! 
//! Memory Alignment Requirements:
//! - All buffers are 64-bit aligned (f64 = 8 bytes)
//! - Batch sizes should be multiples of 2 for f64x2 lanes
//! - Structure-of-Arrays layout enables contiguous SIMD loads

#![no_std]

use core::f64;

// ============================================================================
// LANE CONFIGURATION (tuned for auto-vectorization)
// ============================================================================

/// Number of f64 elements per SIMD vector (f64x2)
const F64_LANES: usize = 2;

/// Minimum batch size for SIMD-optimized paths
const MIN_BATCH_SIZE_SIMD: usize = 4;

/// Alignment requirement for f64 SIMD operations
const ALIGNMENT_F64: usize = 16; // 128-bit alignment for v128

// ============================================================================
// MEMORY LAYOUT CONSTANTS (must match dreamer_kernel_contract.js)
// ============================================================================

/// Control header slots (fixed positions)
const CONTROL_SLOTS: usize = 6;
const META_SLOTS_PER_SERIES: usize = 2;
const FEATURE_COUNT: usize = 6;

/// Feature indices for output buffers
const FEATURE_MEAN_PRICE: usize = 0;
const FEATURE_PRICE_VARIANCE: usize = 1;
const FEATURE_VWAP: usize = 2;
const FEATURE_LATEST_PRICE: usize = 3;
const FEATURE_PRICE_MOMENTUM: usize = 4;
const FEATURE_MEAN_VOLUME: usize = 5;

/// Regime encoding constants
const REGIME_UNKNOWN: f64 = 0.0;
const REGIME_CRAB_CHOP: f64 = 1.0;
const REGIME_BULL_RUSH: f64 = 2.0;
const REGIME_BEAR_CRASH: f64 = 3.0;
const REGIME_STEADY_GROWTH: f64 = 4.0;
const REGIME_VOLATILE_CHOP: f64 = 5.0;

// ============================================================================
// STATIC RESULT BUFFER (aligned for SIMD access)
// ============================================================================

/// Global result buffer - 16 f64 slots (128 bytes)
/// Aligned to 16 bytes for SIMD compatibility
#[repr(C, align(16))]
struct ResultBuffer {
    data: [f64; 16],
}

const RESULT_BUFFER_LEN: usize = 16;

static mut RESULT_BUFFER: ResultBuffer = ResultBuffer { data: [0.0; RESULT_BUFFER_LEN] };

// ============================================================================
// MEMORY LAYOUT COMPUTATION
// ============================================================================

/// Compute buffer layout offsets for Structure-of-Arrays layout.
/// This layout optimizes for contiguous SIMD loads across series.
#[inline(always)]
fn layout_offsets(max_series: usize, samples_per_series: usize) -> (usize, usize, usize, usize) {
    // Control header at start
    let meta_start = CONTROL_SLOTS;
    
    // Metadata section
    let meta_end = meta_start + max_series * META_SLOTS_PER_SERIES;
    
    // Prices section (contiguous per series)
    let prices_start = meta_end;
    let prices_end = prices_start + max_series * samples_per_series;
    
    // Volumes section (contiguous per series)
    let volumes_start = prices_end;
    let volumes_end = volumes_start + max_series * samples_per_series;
    
    // Feature outputs (contiguous per series)
    let feature_start = volumes_end;
    
    (meta_start, prices_start, volumes_start, feature_start)
}

/// Compute buffer size in f64 slots for given configuration
#[inline(always)]
fn compute_buffer_slots(max_series: usize, samples_per_series: usize) -> usize {
    let (_, _, _, feature_start) = layout_offsets(max_series, samples_per_series);
    feature_start + max_series * FEATURE_COUNT
}

// ============================================================================
// POINTER ACCESS HELPERS (inlined for zero-overhead access)
// ============================================================================

/// Get pointer to series metadata (write_index, sample_count)
#[inline(always)]
fn series_meta_ptr(base: *mut f64, series_id: usize, meta_start: usize) -> *mut f64 {
    unsafe { base.add(meta_start + series_id * META_SLOTS_PER_SERIES) }
}

/// Get series metadata values
#[inline(always)]
fn get_series_meta(base: *mut f64, series_id: usize, meta_start: usize) -> (usize, usize) {
    let ptr = series_meta_ptr(base, series_id, meta_start);
    unsafe {
        let write_idx = *ptr as usize;
        let sample_count = *ptr.add(1) as usize;
        (write_idx, sample_count)
    }
}

/// Set series metadata values
#[inline(always)]
fn set_series_meta(
    base: *mut f64,
    series_id: usize,
    meta_start: usize,
    write_idx: usize,
    sample_count: usize
) {
    let ptr = series_meta_ptr(base, series_id, meta_start);
    unsafe {
        *ptr = write_idx as f64;
        *ptr.add(1) = sample_count as f64;
    }
}

/// Get pointer to price buffer for a series
#[inline(always)]
fn price_ptr(base: *mut f64, series_id: usize, prices_start: usize, samples_per_series: usize) -> *mut f64 {
    unsafe { base.add(prices_start + series_id * samples_per_series) }
}

/// Get pointer to volume buffer for a series
#[inline(always)]
fn volume_ptr(base: *mut f64, series_id: usize, volumes_start: usize, samples_per_series: usize) -> *mut f64 {
    unsafe { base.add(volumes_start + series_id * samples_per_series) }
}

/// Get pointer to feature output for a series
#[inline(always)]
fn feature_ptr(base: *mut f64, series_id: usize, feature_start: usize) -> *mut f64 {
    unsafe { base.add(feature_start + series_id * FEATURE_COUNT) }
}

// ============================================================================
// MATH UTILITIES (stable, no_std compatible)
// ============================================================================

/// Square root approximation using Newton-Raphson method
/// Required for no_std stable Rust
#[inline(always)]
fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NAN };
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    let mut guess = x;

    // Newton-Raphson iteration: x_{n+1} = 0.5 * (x_n + S/x_n)
    for _ in 0..10 {
        let prev = guess;
        guess = 0.5 * (guess + x / guess);
        if (guess - prev).abs() < 1e-15 {
            break;
        }
    }

    guess
}

// ============================================================================
// SIMD SHAPE CONSTRAINED BATCH KERNELS
// ============================================================================

/// Batch accumulator structure using interleaved accumulators.
/// This pattern prevents instruction-level data dependencies and
/// allows auto-vectorization to f64x2 SIMD operations.
#[repr(C)]
struct BatchAccumulator {
    sum_0: f64,
    sum_1: f64,
    sum_sq_0: f64,
    sum_sq_1: f64,
    vol_0: f64,
    vol_1: f64,
    notional_0: f64,
    notional_1: f64,
}

impl BatchAccumulator {
    fn new() -> Self {
        BatchAccumulator {
            sum_0: 0.0, sum_1: 0.0,
            sum_sq_0: 0.0, sum_sq_1: 0.0,
            vol_0: 0.0, vol_1: 0.0,
            notional_0: 0.0, notional_1: 0.0,
        }
    }
    
    #[inline(always)]
    fn fold(&self) -> (f64, f64, f64, f64) {
        (
            self.sum_0 + self.sum_1,
            self.sum_sq_0 + self.sum_sq_1,
            self.vol_0 + self.vol_1,
            self.notional_0 + self.notional_1,
        )
    }
}

/// SIMD-friendly batch reduction kernel.
/// Loop structure is designed to auto-vectorize to f64x2 operations.
/// The interleaved accumulator pattern breaks dependencies.
#[inline(always)]
fn batch_reduce_2lane(prices: *const f64, volumes: *const f64, count: usize) -> (f64, f64, f64, f64) {
    let mut acc = BatchAccumulator::new();
    
    // Process in chunks of 2 for optimal f64x2 lane usage
    let chunks = count / F64_LANES;
    let remainder = count % F64_LANES;

    unsafe {
        // Main vectorizable loop - compiler auto-vectorizes this to f64x2
        for i in 0..chunks {
            let idx = i * F64_LANES;
            
            let p0 = *prices.add(idx);
            let p1 = *prices.add(idx + 1);
            let v0 = *volumes.add(idx);
            let v1 = *volumes.add(idx + 1);
            
            acc.sum_0 += p0;
            acc.sum_1 += p1;
            acc.sum_sq_0 += p0 * p0;
            acc.sum_sq_1 += p1 * p1;
            acc.vol_0 += v0;
            acc.vol_1 += v1;
            acc.notional_0 += p0 * v0;
            acc.notional_1 += p1 * v1;
        }
        
        // Handle remainder
        if remainder > 0 {
            let p = *prices.add(count - 1);
            let v = *volumes.add(count - 1);
            acc.sum_0 += p;
            acc.sum_sq_0 += p * p;
            acc.vol_0 += v;
            acc.notional_0 += p * v;
        }
    }
    
    acc.fold()
}

/// Alternative 4-lane accumulator for larger batch sizes
#[inline(always)]
fn batch_reduce_4lane(prices: *const f64, volumes: *const f64, count: usize) -> (f64, f64, f64, f64) {
    // Four independent accumulators for auto-vectorization
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    
    let mut sq0 = 0.0f64;
    let mut sq1 = 0.0f64;
    let mut sq2 = 0.0f64;
    let mut sq3 = 0.0f64;
    
    let mut v0 = 0.0f64;
    let mut v1 = 0.0f64;
    let mut v2 = 0.0f64;
    let mut v3 = 0.0f64;
    
    let mut n0 = 0.0f64;
    let mut n1 = 0.0f64;
    let mut n2 = 0.0f64;
    let mut n3 = 0.0f64;
    
    let chunks = count / 4;
    let remainder_start = chunks * 4;
    
    unsafe {
        // Main loop - compiler sees independent operations, auto-vectorizes
        for i in 0..chunks {
            let base = i * 4;
            
            let p0 = *prices.add(base);
            let p1 = *prices.add(base + 1);
            let p2 = *prices.add(base + 2);
            let p3 = *prices.add(base + 3);
            
            let vol0 = *volumes.add(base);
            let vol1 = *volumes.add(base + 1);
            let vol2 = *volumes.add(base + 2);
            let vol3 = *volumes.add(base + 3);
            
            s0 += p0; s1 += p1; s2 += p2; s3 += p3;
            sq0 += p0 * p0; sq1 += p1 * p1; sq2 += p2 * p2; sq3 += p3 * p3;
            v0 += vol0; v1 += vol1; v2 += vol2; v3 += vol3;
            n0 += p0 * vol0; n1 += p1 * vol1; n2 += p2 * vol2; n3 += p3 * vol3;
        }
        
        // Handle remainder (0-3 elements)
        for i in remainder_start..count {
            let p = *prices.add(i);
            let vol = *volumes.add(i);
            s0 += p;
            sq0 += p * p;
            v0 += vol;
            n0 += p * vol;
        }
    }
    
    (
        s0 + s1 + s2 + s3,
        sq0 + sq1 + sq2 + sq3,
        v0 + v1 + v2 + v3,
        n0 + n1 + n2 + n3,
    )
}

/// Core rolling window feature computation with SIMD-optimized reduction
#[inline(always)]
fn compute_window_features_vectorized(
    prices: *const f64,
    volumes: *const f64,
    write_index: usize,
    sample_count: usize,
    samples_per_series: usize,
    requested_window: usize,
    out: *mut f64,
) {
    let usable_window = sample_count.min(requested_window);
    
    if usable_window == 0 {
        unsafe {
            // Write NaN to all features
            let nan = f64::NAN;
            for i in 0..FEATURE_COUNT {
                *out.add(i) = nan;
            }
        }
        return;
    }
    
    // Calculate window start (handles ring buffer wrap)
    let window_start = if write_index >= usable_window {
        write_index - usable_window
    } else {
        samples_per_series + write_index - usable_window
    };
    
    let first_span = usable_window.min(samples_per_series - window_start);
    let second_span = usable_window - first_span;
    
    // Get boundary prices for momentum
    let oldest_price = unsafe { *prices.add(window_start) };
    let latest_price = if second_span > 0 {
        unsafe { *prices.add(second_span - 1) }
    } else {
        unsafe { *prices.add(window_start + first_span - 1) }
    };
    
    // Choose reduction strategy based on span sizes
    let (sum_price, sum_price_sq, sum_vol, notional) = if first_span >= MIN_BATCH_SIZE_SIMD {
        // Use SIMD-optimized reduction for first span
        let ptr1 = unsafe { prices.add(window_start) };
        let vol1 = unsafe { volumes.add(window_start) };
        
        let (s1, sq1, v1, n1) = if first_span >= 8 {
            batch_reduce_4lane(ptr1, vol1, first_span)
        } else {
            batch_reduce_2lane(ptr1, vol1, first_span)
        };
        
        // Handle second span if present
        if second_span >= MIN_BATCH_SIZE_SIMD {
            let (s2, sq2, v2, n2) = batch_reduce_2lane(prices, volumes, second_span);
            (s1 + s2, sq1 + sq2, v1 + v2, n1 + n2)
        } else if second_span > 0 {
            // Scalar fallback for small second span
            let mut s2 = 0.0;
            let mut sq2 = 0.0;
            let mut v2 = 0.0;
            let mut n2 = 0.0;
            unsafe {
                for i in 0..second_span {
                    let p = *prices.add(i);
                    let vol = *volumes.add(i);
                    s2 += p;
                    sq2 += p * p;
                    v2 += vol;
                    n2 += p * vol;
                }
            }
            (s1 + s2, sq1 + sq2, v1 + v2, n1 + n2)
        } else {
            (s1, sq1, v1, n1)
        }
    } else {
        // Scalar path for small windows
        let mut sum_price = 0.0;
        let mut sum_price_sq = 0.0;
        let mut sum_vol = 0.0;
        let mut notional = 0.0;
        
        unsafe {
            for i in 0..first_span {
                let p = *prices.add(window_start + i);
                let v = *volumes.add(window_start + i);
                sum_price += p;
                sum_price_sq += p * p;
                sum_vol += v;
                notional += p * v;
            }
            
            for i in 0..second_span {
                let p = *prices.add(i);
                let v = *volumes.add(i);
                sum_price += p;
                sum_price_sq += p * p;
                sum_vol += v;
                notional += p * v;
            }
        }
        
        (sum_price, sum_price_sq, sum_vol, notional)
    };
    
    // Compute final features
    let n = usable_window as f64;
    let mean_price = sum_price / n;
    let variance = (sum_price_sq / n) - (mean_price * mean_price);
    // Numerical stability: clamp small negative variances
    let variance = if variance < 0.0 && variance > -1e-12 { 0.0 } else { variance };
    
    let vwap = if sum_vol == 0.0 { f64::NAN } else { notional / sum_vol };
    let price_momentum = latest_price - oldest_price;
    let mean_volume = sum_vol / n;
    
    // Write outputs
    unsafe {
        *out.add(FEATURE_MEAN_PRICE) = mean_price;
        *out.add(FEATURE_PRICE_VARIANCE) = variance;
        *out.add(FEATURE_VWAP) = vwap;
        *out.add(FEATURE_LATEST_PRICE) = latest_price;
        *out.add(FEATURE_PRICE_MOMENTUM) = price_momentum;
        *out.add(FEATURE_MEAN_VOLUME) = mean_volume;
    }
}

// ============================================================================
// PORTFOLIO KERNELS (SIMD-Optimized)
// ============================================================================

/// SIMD accumulator for portfolio deviation calculations
#[repr(C)]
struct PortfolioAccumulator {
    sum_baseline_diff_0: f64,
    sum_baseline_diff_1: f64,
    sum_managed_0: f64,
    sum_managed_1: f64,
    count_declining_0: f64,
    count_declining_1: f64,
}

impl PortfolioAccumulator {
    fn new() -> Self {
        PortfolioAccumulator {
            sum_baseline_diff_0: 0.0,
            sum_baseline_diff_1: 0.0,
            sum_managed_0: 0.0,
            sum_managed_1: 0.0,
            count_declining_0: 0.0,
            count_declining_1: 0.0,
        }
    }
    
    fn fold(&self) -> (f64, f64, f64) {
        (
            self.sum_baseline_diff_0 + self.sum_baseline_diff_1,
            self.sum_managed_0 + self.sum_managed_1,
            self.count_declining_0 + self.count_declining_1,
        )
    }
}

/// Portfolio aggregation kernel with SIMD batch processing
#[inline(always)]
fn compute_portfolio_features_vectorized(
    values: *const f64,
    baselines: *const f64,
    asset_count: usize,
    _cash_balance: f64,
    harvest_trigger: f64,
    rebalance_trigger: f64,
    cp_trigger_asset_percent: f64,
    cp_trigger_min_negative_dev: f64,
    out_deviations: *mut f64,
    out_harvest_candidates: *mut f64,
    out_rebalance_candidates: *mut f64,
) -> (f64, f64, f64, f64, f64) {
    
    let n = asset_count as f64;
    
    // Use interleaved accumulators for auto-vectorization
    let mut acc = PortfolioAccumulator::new();
    
    let chunks = asset_count / 2;
    let remainder = asset_count % 2;
    
    unsafe {
        for i in 0..chunks {
            let idx = i * 2;
            
            let v0 = *values.add(idx);
            let v1 = *values.add(idx + 1);
            let b0 = *baselines.add(idx);
            let b1 = *baselines.add(idx + 1);
            
            // Compute deviations
            let d0 = if b0 > 0.0 { (v0 - b0) / b0 } else { 0.0 };
            let d1 = if b1 > 0.0 { (v1 - b1) / b1 } else { 0.0 };
            
            *out_deviations.add(idx) = d0;
            *out_deviations.add(idx + 1) = d1;
            
            // Harvest candidates
            *out_harvest_candidates.add(idx) = if d0 >= harvest_trigger { 1.0 } else { 0.0 };
            *out_harvest_candidates.add(idx + 1) = if d1 >= harvest_trigger { 1.0 } else { 0.0 };
            
            // Rebalance candidates
            *out_rebalance_candidates.add(idx) = if d0 <= -rebalance_trigger { 1.0 } else { 0.0 };
            *out_rebalance_candidates.add(idx + 1) = if d1 <= -rebalance_trigger { 1.0 } else { 0.0 };
            
            // Accumlate for crash protection
            if b0 > 0.0 {
                acc.sum_baseline_diff_0 += v0 - b0;
                acc.sum_managed_0 += b0;
                if d0 <= cp_trigger_min_negative_dev {
                    acc.count_declining_0 += 1.0;
                }
            }
            
            if b1 > 0.0 {
                acc.sum_baseline_diff_1 += v1 - b1;
                acc.sum_managed_1 += b1;
                if d1 <= cp_trigger_min_negative_dev {
                    acc.count_declining_1 += 1.0;
                }
            }
        }
        
        // Handle remainder
        if remainder > 0 {
            let idx = asset_count - 1;
            let v = *values.add(idx);
            let b = *baselines.add(idx);
            let d = if b > 0.0 { (v - b) / b } else { 0.0 };
            
            *out_deviations.add(idx) = d;
            *out_harvest_candidates.add(idx) = if d >= harvest_trigger { 1.0 } else { 0.0 };
            *out_rebalance_candidates.add(idx) = if d <= -rebalance_trigger { 1.0 } else { 0.0 };
            
            if b > 0.0 {
                acc.sum_baseline_diff_0 += v - b;
                acc.sum_managed_0 += b;
                if d <= cp_trigger_min_negative_dev {
                    acc.count_declining_0 += 1.0;
                }
            }
        }
    }
    
    let (total_baseline_diff, total_managed_baseline, declining_count) = acc.fold();
    
    let deviation_percent = if total_managed_baseline > 0.0 {
        (total_baseline_diff / total_managed_baseline) * 100.0
    } else {
        0.0
    };
    
    let percent_declining = declining_count / n;
    let crash_active = if total_managed_baseline > 0.0 && percent_declining >= cp_trigger_asset_percent {
        1.0
    } else {
        0.0
    };
    
    (deviation_percent, crash_active, declining_count, total_managed_baseline, total_baseline_diff)
}

// ============================================================================
// REGIME KERNEL
// ============================================================================

/// Regime detection with SIMD-optimized statistics
#[inline(always)]
fn compute_regime_vectorized(
    history: *const f64,
    history_len: usize,
    current_price: f64,
    start_price: f64,
) -> (f64, f64, f64, f64) {
    
    if history_len < 50 || start_price <= 0.0 {
        return (REGIME_UNKNOWN, 0.0, 0.0, 0.0);
    }
    
    let roi = (current_price - start_price) / start_price;
    
    // Compute mean using SIMD-optimized reduction
    let sum = if history_len >= 8 {
        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        let mut s3 = 0.0f64;
        
        let chunks = history_len / 4;
        let rem_start = chunks * 4;
        
        unsafe {
            for i in 0..chunks {
                let base = i * 4;
                s0 += *history.add(base);
                s1 += *history.add(base + 1);
                s2 += *history.add(base + 2);
                s3 += *history.add(base + 3);
            }
            
            for i in rem_start..history_len {
                s0 += *history.add(i);
            }
        }
        
        s0 + s1 + s2 + s3
    } else {
        let mut s = 0.0;
        unsafe {
            for i in 0..history_len {
                s += *history.add(i);
            }
        }
        s
    };
    
    let mean = sum / (history_len as f64);
    
    // Compute variance (second pass)
    let sum_sq_diff = {
        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        let mut s3 = 0.0f64;
        
        let chunks = history_len / 4;
        let rem_start = chunks * 4;
        
        unsafe {
            for i in 0..chunks {
                let base = i * 4;
                let d0 = *history.add(base) - mean;
                let d1 = *history.add(base + 1) - mean;
                let d2 = *history.add(base + 2) - mean;
                let d3 = *history.add(base + 3) - mean;
                
                s0 += d0 * d0;
                s1 += d1 * d1;
                s2 += d2 * d2;
                s3 += d3 * d3;
            }
            
            for i in rem_start..history_len {
                let d = *history.add(i) - mean;
                s0 += d * d;
            }
        }
        
        s0 + s1 + s2 + s3
    };
    
    let variance = sum_sq_diff / (history_len as f64);
    let volatility = if mean > 0.0 { sqrt(variance) / mean } else { 0.0 };
    
    // Classify regime
    let regime = if roi > 0.05 && volatility > 0.02 {
        REGIME_BULL_RUSH
    } else if roi < -0.05 && volatility > 0.02 {
        REGIME_BEAR_CRASH
    } else if roi > 0.02 && volatility < 0.01 {
        REGIME_STEADY_GROWTH
    } else if volatility > 0.05 {
        REGIME_VOLATILE_CHOP
    } else {
        REGIME_CRAB_CHOP
    };
    
    (regime, roi, volatility, mean)
}

// ============================================================================
// DEFECT SCAN KERNEL
// ============================================================================

/// Shadow defect scan with ring buffer support
#[inline(always)]
fn scan_for_defects_scalar(
    prices: *const f64,
    price_count: usize,
    rebalance_trigger: f64,
    crash_threshold: f64,
) -> (f64, f64, f64) {
    
    if price_count < 2 {
        return (0.0, 0.0, 0.0);
    }
    
    let mut max_price: f64;
    let mut is_defective: f64 = 0.0;
    let mut max_drawdown: f64 = 0.0;
    let mut trigger_hits: f64 = 0.0;
    
    unsafe {
        max_price = *prices;
        
        for i in 0..price_count {
            let current_price = *prices.add(i);
            
            if current_price > max_price {
                max_price = current_price;
            }
            
            let deviation = (current_price - max_price) / max_price;
            
            if deviation < rebalance_trigger {
                trigger_hits += 1.0;
                
                // Look ahead for crash
                let mut min_future_price = current_price;
                for j in (i + 1)..price_count {
                    let future_price = *prices.add(j);
                    if future_price < min_future_price {
                        min_future_price = future_price;
                    }
                }
                
                let subsequent_drop = (min_future_price - current_price) / current_price;
                
                if subsequent_drop < -crash_threshold {
                    is_defective = 1.0;
                    let drawdown = -subsequent_drop;
                    if drawdown > max_drawdown {
                        max_drawdown = drawdown;
                    }
                }
            }
        }
    }
    
    (is_defective, max_drawdown, trigger_hits)
}

// ============================================================================
// WASM EXPORT INTERFACE
// ============================================================================

/// Export: Get required buffer size in bytes
#[no_mangle]
pub extern "C" fn get_buffer_size(max_series: usize, samples_per_series: usize) -> usize {
    compute_buffer_slots(max_series, samples_per_series) * 8
}

/// Export: Get alignment requirement in bytes
#[no_mangle]
pub extern "C" fn get_alignment() -> usize {
    ALIGNMENT_F64
}

/// Export: Get feature count
#[no_mangle]
pub extern "C" fn get_feature_count() -> usize {
    FEATURE_COUNT
}

/// Export: Get ABI version
#[no_mangle]
pub extern "C" fn get_abi_version() -> usize {
    100 // 1.0.0
}

/// Export: Compute features for a single series
#[no_mangle]
pub extern "C" fn compute_features(
    arena_ptr: *mut f64,
    series_id: usize,
    window_size: usize,
    max_series: usize,
    samples_per_series: usize,
) -> usize {
    let (meta_start, prices_start, volumes_start, feature_start) = 
        layout_offsets(max_series, samples_per_series);
    
    let (write_idx, sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
    
    let prices = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
    let volumes = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
    let features = feature_ptr(arena_ptr, series_id, feature_start);
    
    compute_window_features_vectorized(
        prices,
        volumes,
        write_idx,
        sample_count,
        samples_per_series,
        window_size,
        features,
    );
    
    series_id
}

/// Export: Batch compute features for multiple series
#[no_mangle]
pub extern "C" fn compute_features_batch(
    arena_ptr: *mut f64,
    series_ids_ptr: *const usize,
    window_sizes_ptr: *const usize,
    batch_count: usize,
    max_series: usize,
    samples_per_series: usize,
) -> usize {
    let (meta_start, prices_start, volumes_start, feature_start) = 
        layout_offsets(max_series, samples_per_series);
    
    // Process batch with SIMD-optimized inner loop
    unsafe {
        for i in 0..batch_count {
            let series_id = *series_ids_ptr.add(i);
            let window_size = *window_sizes_ptr.add(i);
            
            let (write_idx, sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
            
            let prices = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
            let volumes = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
            let features = feature_ptr(arena_ptr, series_id, feature_start);
            
            compute_window_features_vectorized(
                prices,
                volumes,
                write_idx,
                sample_count,
                samples_per_series,
                window_size,
                features,
            );
        }
    }
    
    batch_count
}

/// Export: Batch compute features with fixed window (optimized path)
#[no_mangle]
pub extern "C" fn compute_features_batch_fixed_window(
    arena_ptr: *mut f64,
    series_ids_ptr: *const usize,
    batch_count: usize,
    window_size: usize,
    max_series: usize,
    samples_per_series: usize,
) -> usize {
    let (meta_start, prices_start, volumes_start, feature_start) = 
        layout_offsets(max_series, samples_per_series);
    
    unsafe {
        for i in 0..batch_count {
            let series_id = *series_ids_ptr.add(i);
            
            let (write_idx, sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
            
            let prices = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
            let volumes = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
            let features = feature_ptr(arena_ptr, series_id, feature_start);
            
            compute_window_features_vectorized(
                prices,
                volumes,
                write_idx,
                sample_count,
                samples_per_series,
                window_size,
                features,
            );
        }
    }
    
    batch_count
}

/// Export: Compute features for a single series into an explicit output buffer
#[no_mangle]
pub extern "C" fn compute_features_into(
    arena_ptr: *mut f64,
    series_id: usize,
    window_size: usize,
    max_series: usize,
    samples_per_series: usize,
    out_ptr: *mut f64,
) -> usize {
    let (meta_start, prices_start, volumes_start, _) =
        layout_offsets(max_series, samples_per_series);

    let (write_idx, sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
    let prices = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
    let volumes = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);

    compute_window_features_vectorized(
        prices,
        volumes,
        write_idx,
        sample_count,
        samples_per_series,
        window_size,
        out_ptr,
    );

    FEATURE_COUNT
}

/// Export: Batch compute features for multiple series into an explicit output slab
#[no_mangle]
pub extern "C" fn compute_features_batch_into(
    arena_ptr: *mut f64,
    series_ids_ptr: *const usize,
    window_sizes_ptr: *const usize,
    batch_count: usize,
    max_series: usize,
    samples_per_series: usize,
    out_ptr: *mut f64,
) -> usize {
    let (meta_start, prices_start, volumes_start, _) =
        layout_offsets(max_series, samples_per_series);

    unsafe {
        for i in 0..batch_count {
            let series_id = *series_ids_ptr.add(i);
            let window_size = *window_sizes_ptr.add(i);
            let (write_idx, sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
            let prices = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
            let volumes = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
            let out_row = out_ptr.add(i * FEATURE_COUNT);

            compute_window_features_vectorized(
                prices,
                volumes,
                write_idx,
                sample_count,
                samples_per_series,
                window_size,
                out_row,
            );
        }
    }

    batch_count
}

/// Export: Batch compute features with one fixed window into an explicit output slab
#[no_mangle]
pub extern "C" fn compute_features_batch_fixed_window_into(
    arena_ptr: *mut f64,
    series_ids_ptr: *const usize,
    batch_count: usize,
    window_size: usize,
    max_series: usize,
    samples_per_series: usize,
    out_ptr: *mut f64,
) -> usize {
    let (meta_start, prices_start, volumes_start, _) =
        layout_offsets(max_series, samples_per_series);

    unsafe {
        for i in 0..batch_count {
            let series_id = *series_ids_ptr.add(i);
            let (write_idx, sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
            let prices = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
            let volumes = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
            let out_row = out_ptr.add(i * FEATURE_COUNT);

            compute_window_features_vectorized(
                prices,
                volumes,
                write_idx,
                sample_count,
                samples_per_series,
                window_size,
                out_row,
            );
        }
    }

    batch_count
}

/// Export: Compute portfolio features
#[no_mangle]
pub extern "C" fn compute_portfolio(
    arena_ptr: *mut f64,
    asset_count: usize,
    cash_balance: f64,
    harvest_trigger: f64,
    rebalance_trigger: f64,
    cp_trigger_asset_percent: f64,
    cp_trigger_min_negative_dev: f64,
) -> usize {
    // Output buffers follow feature section
    let (_, _, _, feature_start) = layout_offsets(256, 1024);
    let output_offset = feature_start + 256 * FEATURE_COUNT;
    
    let values = unsafe { arena_ptr.add(output_offset) };
    let baselines = unsafe { arena_ptr.add(output_offset + asset_count) };
    let out_deviations = unsafe { arena_ptr.add(output_offset + 2 * asset_count) };
    let out_harvest = unsafe { arena_ptr.add(output_offset + 3 * asset_count) };
    let out_rebalance = unsafe { arena_ptr.add(output_offset + 4 * asset_count) };
    
    let (deviation_pct, crash_active, declining, managed_baseline, baseline_diff) = 
        compute_portfolio_features_vectorized(
            values,
            baselines,
            asset_count,
            cash_balance,
            harvest_trigger,
            rebalance_trigger,
            cp_trigger_asset_percent,
            cp_trigger_min_negative_dev,
            out_deviations,
            out_harvest,
            out_rebalance,
        );
    
    unsafe {
        RESULT_BUFFER.data[0] = deviation_pct;
        RESULT_BUFFER.data[1] = crash_active;
        RESULT_BUFFER.data[2] = declining;
        RESULT_BUFFER.data[3] = managed_baseline;
        RESULT_BUFFER.data[4] = baseline_diff;
    }
    
    0 // Success
}

/// Export: Compute portfolio features using explicit input and output pointers
#[no_mangle]
pub extern "C" fn compute_portfolio_into(
    values_ptr: *const f64,
    baselines_ptr: *const f64,
    asset_count: usize,
    cash_balance: f64,
    harvest_trigger: f64,
    rebalance_trigger: f64,
    cp_trigger_asset_percent: f64,
    cp_trigger_min_negative_dev: f64,
    aggregate_out_ptr: *mut f64,
    deviations_out_ptr: *mut f64,
    harvest_out_ptr: *mut f64,
    rebalance_out_ptr: *mut f64,
) -> usize {
    let (deviation_pct, crash_active, declining, managed_baseline, baseline_diff) =
        compute_portfolio_features_vectorized(
            values_ptr,
            baselines_ptr,
            asset_count,
            cash_balance,
            harvest_trigger,
            rebalance_trigger,
            cp_trigger_asset_percent,
            cp_trigger_min_negative_dev,
            deviations_out_ptr,
            harvest_out_ptr,
            rebalance_out_ptr,
        );

    unsafe {
        *aggregate_out_ptr.add(0) = deviation_pct;
        *aggregate_out_ptr.add(1) = crash_active;
        *aggregate_out_ptr.add(2) = declining;
        *aggregate_out_ptr.add(3) = managed_baseline;
        *aggregate_out_ptr.add(4) = baseline_diff;
    }

    0
}

/// Export: Scan for defects into an explicit output buffer
#[no_mangle]
pub extern "C" fn scan_defects_into(
    prices_ptr: *const f64,
    price_count: usize,
    rebalance_trigger: f64,
    crash_threshold: f64,
    out_ptr: *mut f64,
) -> usize {
    let (is_defective, max_drawdown, trigger_hits) =
        scan_for_defects_scalar(prices_ptr, price_count, rebalance_trigger, crash_threshold);

    unsafe {
        *out_ptr.add(0) = is_defective;
        *out_ptr.add(1) = max_drawdown;
        *out_ptr.add(2) = trigger_hits;
    }

    0
}

/// Export: Compute regime statistics into an explicit output buffer
#[no_mangle]
pub extern "C" fn compute_regime_into(
    history_ptr: *const f64,
    history_len: usize,
    current_price: f64,
    start_price: f64,
    out_ptr: *mut f64,
) -> usize {
    let (regime, roi, volatility, mean) =
        compute_regime_vectorized(history_ptr, history_len, current_price, start_price);

    unsafe {
        *out_ptr.add(0) = regime;
        *out_ptr.add(1) = roi;
        *out_ptr.add(2) = volatility;
        *out_ptr.add(3) = mean;
    }

    0
}

/// Export: Get result by index
#[no_mangle]
pub extern "C" fn get_f64_result(index: usize) -> f64 {
    unsafe {
        if index < RESULT_BUFFER_LEN {
            RESULT_BUFFER.data[index]
        } else {
            f64::NAN
        }
    }
}

/// Export: Scan for defects
#[no_mangle]
pub extern "C" fn scan_defects(
    prices_ptr: *const f64,
    price_count: usize,
    rebalance_trigger: f64,
    crash_threshold: f64,
) -> usize {
    let (is_defective, max_drawdown, trigger_hits) = scan_for_defects_scalar(
        prices_ptr,
        price_count,
        rebalance_trigger,
        crash_threshold,
    );
    
    unsafe {
        RESULT_BUFFER.data[0] = is_defective;
        RESULT_BUFFER.data[1] = max_drawdown;
        RESULT_BUFFER.data[2] = trigger_hits;
    }
    
    if is_defective > 0.5 { 1 } else { 0 }
}

/// Export: Compute regime
#[no_mangle]
pub extern "C" fn compute_regime_export(
    history_ptr: *const f64,
    history_len: usize,
    current_price: f64,
    start_price: f64,
) -> f64 {
    let (regime, roi, volatility, mean) = compute_regime_vectorized(
        history_ptr,
        history_len,
        current_price,
        start_price,
    );
    
    unsafe {
        RESULT_BUFFER.data[0] = regime;
        RESULT_BUFFER.data[1] = roi;
        RESULT_BUFFER.data[2] = volatility;
        RESULT_BUFFER.data[3] = mean;
    }
    
    regime
}

/// Export: Ingest a single tick into ring buffer
#[no_mangle]
pub extern "C" fn ingest_tick(
    arena_ptr: *mut f64,
    series_id: usize,
    price: f64,
    volume: f64,
    max_series: usize,
    samples_per_series: usize,
) -> usize {
    let (meta_start, prices_start, volumes_start, _) = 
        layout_offsets(max_series, samples_per_series);
    
    let (write_idx, sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
    
    unsafe {
        let price_buf = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
        let volume_buf = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
        
        *price_buf.add(write_idx) = price;
        *volume_buf.add(write_idx) = volume;
    }
    
    let new_write_idx = (write_idx + 1) % samples_per_series;
    let new_sample_count = if sample_count < samples_per_series {
        sample_count + 1
    } else {
        sample_count
    };
    
    set_series_meta(arena_ptr, series_id, meta_start, new_write_idx, new_sample_count);
    
    unsafe {
        *arena_ptr.add(3) = series_id as f64;
        *arena_ptr.add(4) = price;
        *arena_ptr.add(5) = volume;
    }
    
    new_sample_count
}

/// Export: Batch ingest ticks
#[no_mangle]
pub extern "C" fn ingest_batch(
    arena_ptr: *mut f64,
    series_id: usize,
    prices_ptr: *const f64,
    volumes_ptr: *const f64,
    count: usize,
    max_series: usize,
    samples_per_series: usize,
) -> usize {
    let (meta_start, prices_start, volumes_start, _) = 
        layout_offsets(max_series, samples_per_series);
    
    let (mut write_idx, mut sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
    let price_buf = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
    let volume_buf = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
    
    unsafe {
        for i in 0..count {
            let price = *prices_ptr.add(i);
            let volume = *volumes_ptr.add(i);
            
            *price_buf.add(write_idx) = price;
            *volume_buf.add(write_idx) = volume;
            
            write_idx = (write_idx + 1) % samples_per_series;
            if sample_count < samples_per_series {
                sample_count += 1;
            }
        }
        
        set_series_meta(arena_ptr, series_id, meta_start, write_idx, sample_count);
    }
    
    sample_count
}

// ============================================================================
// PANIC HANDLER (required for no_std)
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
