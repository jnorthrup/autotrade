//! Dreamer WASM Kernel - Rust Implementation
//! 
//! This module implements the exact numeric hot paths promoted into WASM.
//! All kernel inputs, outputs, and 2-degree-of-contact invariants are in f64.
//! 
//! Build: cargo build --target wasm32-unknown-unknown --release
//! SIMD:  -C target-feature=+simd128

#![no_std]
#![feature(simd_128)]

use core::f64;

// ============================================================================
// Memory Layout Constants (must match dreamer_kernel_contract.js)
// ============================================================================

const CONTROL_SLOTS: usize = 6;
const META_SLOTS_PER_SERIES: usize = 2;
const FEATURE_COUNT: usize = 6;

// Feature indices
const FEATURE_MEAN_PRICE: usize = 0;
const FEATURE_PRICE_VARIANCE: usize = 1;
const FEATURE_VWAP: usize = 2;
const FEATURE_LATEST_PRICE: usize = 3;
const FEATURE_PRICE_MOMENTUM: usize = 4;
const FEATURE_MEAN_VOLUME: usize = 5;

// ============================================================================
// Linear Memory Interface
// ============================================================================

/// Compute buffer layout offsets
#[inline(always)]
fn layout_offsets(max_series: usize, samples_per_series: usize) -> (usize, usize, usize, usize) {
    let series_meta_start = CONTROL_SLOTS;
    let series_meta_end = series_meta_start + max_series * META_SLOTS_PER_SERIES;
    let prices_start = series_meta_end;
    let prices_end = prices_start + max_series * samples_per_series;
    let volumes_start = prices_end;
    let volumes_end = volumes_start + max_series * samples_per_series;
    let feature_start = volumes_end;
    
    (series_meta_start, prices_start, volumes_start, feature_start)
}

/// Get pointer to series metadata
#[inline(always)]
fn series_meta_ptr(base: *mut f64, series_id: usize, meta_start: usize) -> *mut f64 {
    unsafe { base.add(meta_start + series_id * META_SLOTS_PER_SERIES) }
}

/// Get write index and sample count for a series
#[inline(always)]
fn get_series_meta(base: *mut f64, series_id: usize, meta_start: usize) -> (usize, usize) {
    let ptr = series_meta_ptr(base, series_id, meta_start);
    unsafe {
        let write_idx = *ptr as usize;
        let sample_count = *ptr.add(1) as usize;
        (write_idx, sample_count)
    }
}

/// Set series metadata
#[inline(always)]
fn set_series_meta(base: *mut f64, series_id: usize, meta_start: usize, write_idx: usize, sample_count: usize) {
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
// Core Kernel: Rolling Window Features
// ============================================================================

/// Compute rolling window features for a single series
/// This is a hot path - all operations are on f64 vectors
#[inline(always)]
fn compute_window_features(
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
        // Fill with NaN
        unsafe {
            for i in 0..FEATURE_COUNT {
                *out.add(i) = f64::NAN;
            }
        }
        return;
    }
    
    // Calculate window start (may wrap)
    let window_start = if write_index >= usable_window {
        write_index - usable_window
    } else {
        samples_per_series + write_index - usable_window
    };
    
    let first_span = usable_window.min(samples_per_series - window_start);
    let second_span = usable_window - first_span;
    
    // Accumulate statistics
    let mut sum_price: f64 = 0.0;
    let mut sum_price_sq: f64 = 0.0;
    let mut sum_volume: f64 = 0.0;
    let mut notional: f64 = 0.0;
    let mut oldest_price: f64 = 0.0;
    let mut latest_price: f64 = 0.0;
    
    // First span (contiguous tail)
    unsafe {
        oldest_price = *prices.add(window_start);
        
        for i in 0..first_span {
            let price = *prices.add(window_start + i);
            let volume = *volumes.add(window_start + i);
            
            latest_price = price;
            sum_price += price;
            sum_price_sq += price * price;
            sum_volume += volume;
            notional += price * volume;
        }
    }
    
    // Second span (wrapped head, if needed)
    if second_span > 0 {
        unsafe {
            for i in 0..second_span {
                let price = *prices.add(i);
                let volume = *volumes.add(i);
                
                latest_price = price;
                sum_price += price;
                sum_price_sq += price * price;
                sum_volume += volume;
                notional += price * volume;
            }
        }
    }
    
    // Compute features
    let n = usable_window as f64;
    let mean_price = sum_price / n;
    let variance = (sum_price_sq / n) - (mean_price * mean_price);
    // Numerical stability: clamp negative variance near zero
    let variance = if variance < 0.0 && variance > -1e-12 { 0.0 } else { variance };
    
    let vwap = if sum_volume == 0.0 { f64::NAN } else { notional / sum_volume };
    let price_momentum = latest_price - oldest_price;
    let mean_volume = sum_volume / n;
    
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
// Core Kernel: Portfolio Features
// ============================================================================

/// Portfolio aggregation kernel
/// Inputs: arrays of values, baselines, etc.
/// Outputs: deviation, crash protection flags
#[inline(always)]
fn compute_portfolio_features(
    values: *const f64,
    baselines: *const f64,
    asset_count: usize,
    cash_balance: f64,
    harvest_trigger: f64,
    rebalance_trigger: f64,
    cp_trigger_asset_percent: f64,
    cp_trigger_min_negative_dev: f64,
    out_deviations: *mut f64,
    out_harvest_candidates: *mut f64,
    out_rebalance_candidates: *mut f64,
) -> (f64, f64, f64, f64, f64) { // Returns: dev%, crash_active, declining_count, managed_baseline, baseline_diff
    
    let mut total_baseline_diff: f64 = 0.0;
    let mut total_managed_baseline: f64 = 0.0;
    let mut declining_count: f64 = 0.0;
    let n = asset_count as f64;
    
    unsafe {
        for i in 0..asset_count {
            let value = *values.add(i);
            let baseline = *baselines.add(i);
            
            let deviation = if baseline > 0.0 {
                (value - baseline) / baseline
            } else {
                0.0
            };
            
            *out_deviations.add(i) = deviation;
            
            // Harvest candidate: deviation >= trigger
            if deviation >= harvest_trigger {
                *out_harvest_candidates.add(i) = 1.0;
            } else {
                *out_harvest_candidates.add(i) = 0.0;
            }
            
            // Rebalance candidate: deviation <= -trigger (negative deviation beyond threshold)
            if deviation <= -rebalance_trigger {
                *out_rebalance_candidates.add(i) = 1.0;
            } else {
                *out_rebalance_candidates.add(i) = 0.0;
            }
            
            // Crash protection counting
            if baseline > 0.0 {
                total_baseline_diff += (value - baseline);
                total_managed_baseline += baseline;
                
                if deviation <= cp_trigger_min_negative_dev {
                    declining_count += 1.0;
                }
            }
        }
    }
    
    let deviation_percent = if total_managed_baseline > 0.0 {
        (total_baseline_diff / total_managed_baseline) * 100.0
    } else {
        0.0
    };
    
    let crash_active = if total_managed_baseline > 0.0 {
        let percent_declining = declining_count / n;
        if percent_declining >= cp_trigger_asset_percent {
            1.0
        } else {
            0.0
        }
    } else {
        0.0
    };
    
    (deviation_percent, crash_active, declining_count, total_managed_baseline, total_baseline_diff)
}

// ============================================================================
// Core Kernel: Defect Scan
// ============================================================================

/// Shadow defect scan kernel
/// Detects if a rebalancing strategy would have triggered a bad trade
#[inline(always)]
fn scan_for_defects(
    prices: *const f64,
    price_count: usize,
    rebalance_trigger: f64,
    crash_threshold: f64,
) -> (f64, f64, f64) { // Returns: is_defective, max_drawdown, trigger_hits
    
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
            
            // Check if trigger condition is met
            // Trigger: (price - max) / max < rebalance_trigger (which is negative)
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
// Core Kernel: Regime Detection
// ============================================================================

/// Regime encoding (must match contract)
const REGIME_UNKNOWN: f64 = 0.0;
const REGIME_CRAB_CHOP: f64 = 1.0;
const REGIME_BULL_RUSH: f64 = 2.0;
const REGIME_BEAR_CRASH: f64 = 3.0;
const REGIME_STEADY_GROWTH: f64 = 4.0;
const REGIME_VOLATILE_CHOP: f64 = 5.0;

/// Compute market regime from price history
#[inline(always)]
fn compute_regime(
    history: *const f64,
    history_len: usize,
    current_price: f64,
    start_price: f64,
) -> (f64, f64, f64, f64) { // Returns: regime, roi, volatility, mean
    
    if history_len < 50 || start_price <= 0.0 {
        return (REGIME_UNKNOWN, 0.0, 0.0, 0.0);
    }
    
    let roi = (current_price - start_price) / start_price;
    
    // Compute mean
    let mut sum: f64 = 0.0;
    unsafe {
        for i in 0..history_len {
            sum += *history.add(i);
        }
    }
    let mean = sum / (history_len as f64);
    
    // Compute variance
    let mut sum_sq_diff: f64 = 0.0;
    unsafe {
        for i in 0..history_len {
            let diff = *history.add(i) - mean;
            sum_sq_diff += diff * diff;
        }
    }
    let variance = sum_sq_diff / (history_len as f64);
    let volatility = if mean > 0.0 { variance.sqrt() / mean } else { 0.0 };
    
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
// WASM Export Interface
// ============================================================================

/// Global result buffer for simple returns
static mut RESULT_BUFFER: [f64; 16] = [0.0; 16];

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
    
    compute_window_features(
        prices,
        volumes,
        write_idx,
        sample_count,
        samples_per_series,
        window_size,
        features,
    );
    
    series_id // Return series_id as success indicator
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
    
    unsafe {
        for i in 0..batch_count {
            let series_id = *series_ids_ptr.add(i);
            let window_size = *window_sizes_ptr.add(i);
            
            let (write_idx, sample_count) = get_series_meta(arena_ptr, series_id, meta_start);
            
            let prices = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
            let volumes = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
            let features = feature_ptr(arena_ptr, series_id, feature_start);
            
            compute_window_features(
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
    // Layout requires additional output buffers after feature section
    // For simplicity, we'll use fixed offsets in arena
    let output_offset = 1024; // Reserve space for outputs
    
    let values = unsafe { arena_ptr.add(output_offset) };
    let baselines = unsafe { arena_ptr.add(output_offset + asset_count) };
    let out_deviations = unsafe { arena_ptr.add(output_offset + 2 * asset_count) };
    let out_harvest = unsafe { arena_ptr.add(output_offset + 3 * asset_count) };
    let out_rebalance = unsafe { arena_ptr.add(output_offset + 4 * asset_count) };
    
    let (deviation_pct, crash_active, declining, managed_baseline, baseline_diff) = 
        compute_portfolio_features(
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
        RESULT_BUFFER[0] = deviation_pct;
        RESULT_BUFFER[1] = crash_active;
        RESULT_BUFFER[2] = declining;
        RESULT_BUFFER[3] = managed_baseline;
        RESULT_BUFFER[4] = baseline_diff;
    }
    
    0 // Success
}

/// Export: Get f64 result by index
#[no_mangle]
pub extern "C" fn get_f64_result(index: usize) -> f64 {
    unsafe {
        if index < RESULT_BUFFER.len() {
            RESULT_BUFFER[index]
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
    let (is_defective, max_drawdown, trigger_hits) = scan_for_defects(
        prices_ptr,
        price_count,
        rebalance_trigger,
        crash_threshold,
    );
    
    unsafe {
        RESULT_BUFFER[0] = is_defective;
        RESULT_BUFFER[1] = max_drawdown;
        RESULT_BUFFER[2] = trigger_hits;
    }
    
    if is_defective > 0.5 {
        1
    } else {
        0
    }
}

/// Export: Compute regime
#[no_mangle]
pub extern "C" fn compute_regime_export(
    history_ptr: *const f64,
    history_len: usize,
    current_price: f64,
    start_price: f64,
) -> f64 {
    let (regime, roi, volatility, mean) = compute_regime(
        history_ptr,
        history_len,
        current_price,
        start_price,
    );
    
    unsafe {
        RESULT_BUFFER[0] = regime;
        RESULT_BUFFER[1] = roi;
        RESULT_BUFFER[2] = volatility;
        RESULT_BUFFER[3] = mean;
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
    
    // Write to buffers
    unsafe {
        let price_buf = price_ptr(arena_ptr, series_id, prices_start, samples_per_series);
        let volume_buf = volume_ptr(arena_ptr, series_id, volumes_start, samples_per_series);
        
        *price_buf.add(write_idx) = price;
        *volume_buf.add(write_idx) = volume;
    }
    
    // Update metadata
    let new_write_idx = (write_idx + 1) % samples_per_series;
    let new_sample_count = if sample_count < samples_per_series {
        sample_count + 1
    } else {
        sample_count
    };
    
    set_series_meta(arena_ptr, series_id, meta_start, new_write_idx, new_sample_count);
    
    // Update control header
    unsafe {
        *arena_ptr.add(3) = series_id as f64;
        *arena_ptr.add(4) = price;
        *arena_ptr.add(5) = volume;
    }
    
    new_sample_count
}

// ============================================================================
// Panic Handler (required for no_std)
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
