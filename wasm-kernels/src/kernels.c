// Core numeric kernels for financial indicators
// Optimized for WASM SIMD auto-vectorization with strict IEEE-754 fidelity

#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#define WASM_EXPORT __attribute__((visibility("default")))
#define ALIGN_32 __attribute__((aligned(32)))

// Simple Moving Average (SMA)
WASM_EXPORT void sma_kernel(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    const double inv_period = 1.0 / (double)period;
    double sum = 0.0;
    
    // Initialize first window
    for (int i = 0; i < period; i++) {
        sum += prices[i];
    }
    output[period - 1] = sum * inv_period;
    
    // Rolling window calculation - no branches in inner loop
    for (int i = period; i < length; i++) {
        sum = sum - prices[i - period] + prices[i];
        output[i] = sum * inv_period;
    }
}

// Exponential Moving Average (EMA)
WASM_EXPORT void ema_kernel(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    const double alpha = 2.0 / (period + 1.0);
    const double one_minus_alpha = 1.0 - alpha;
    
    // First EMA value is the first price
    output[0] = prices[0];
    
    // EMA calculation - monomorphic loop
    for (int i = 1; i < length; i++) {
        output[i] = alpha * prices[i] + one_minus_alpha * output[i - 1];
    }
}

// True Range and Average True Range (ATR)
WASM_EXPORT void atr_kernel(const double* high, const double* low, const double* close, 
                           double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    // Calculate True Range for each bar
    double tr_values[4096]; // Fixed size for SIMD optimization
    if (length > 4096) return; // Safety check
    
    tr_values[0] = high[0] - low[0]; // First TR is just high-low
    
    for (int i = 1; i < length; i++) {
        const double hl = high[i] - low[i];
        const double hc = fabs(high[i] - close[i - 1]);
        const double lc = fabs(low[i] - close[i - 1]);
        
        // Use fmax for SIMD-friendly selection
        const double max1 = fmax(hl, hc);
        tr_values[i] = fmax(max1, lc);
    }
    
    // Calculate ATR using Wilder's smoothing
    const double alpha = 1.0 / period;
    const double one_minus_alpha = 1.0 - alpha;
    
    // First ATR is simple average of first period TR values
    double atr_sum = 0.0;
    for (int i = 0; i < period; i++) {
        atr_sum += tr_values[i];
    }
    output[period - 1] = atr_sum / period;
    
    // Wilder's smoothing for remaining values
    for (int i = period; i < length; i++) {
        output[i] = alpha * tr_values[i] + one_minus_alpha * output[i - 1];
    }
}

// Relative Strength Index (RSI)
WASM_EXPORT void rsi_kernel(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    double gains[4096];
    double losses[4096];
    
    if (length > 4096) return; // Safety check
    
    // Calculate price changes
    for (int i = 1; i < length; i++) {
        const double change = prices[i] - prices[i - 1];
        gains[i] = change > 0.0 ? change : 0.0;
        losses[i] = change < 0.0 ? -change : 0.0;
    }
    
    // Initialize first average gain and loss
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    
    for (int i = 1; i <= period; i++) {
        avg_gain += gains[i];
        avg_loss += losses[i];
    }
    avg_gain /= period;
    avg_loss /= period;
    
    const double alpha = 1.0 / period;
    const double one_minus_alpha = 1.0 - alpha;
    
    // First RSI value
    if (avg_loss > 0.0) {
        const double rs = avg_gain / avg_loss;
        output[period] = 100.0 - (100.0 / (1.0 + rs));
    } else {
        output[period] = 100.0;
    }
    
    // Calculate remaining RSI values
    for (int i = period + 1; i < length; i++) {
        avg_gain = alpha * gains[i] + one_minus_alpha * avg_gain;
        avg_loss = alpha * losses[i] + one_minus_alpha * avg_loss;
        
        if (avg_loss > 0.0) {
            const double rs = avg_gain / avg_loss;
            output[i] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            output[i] = 100.0;
        }
    }
}

// MACD (Moving Average Convergence Divergence)
WASM_EXPORT void macd_kernel(const double* prices, double* macd_line, double* signal_line, 
                            double* histogram, const int length, 
                            const int fast_period, const int slow_period, const int signal_period) {
    if (fast_period >= slow_period) return;
    if (signal_period <= 0) return;
    
    double fast_ema[4096];
    double slow_ema[4096];
    
    if (length > 4096) return; // Safety check
    
    // Calculate fast and slow EMAs
    const double fast_alpha = 2.0 / (fast_period + 1.0);
    const double slow_alpha = 2.0 / (slow_period + 1.0);
    const double one_minus_fast = 1.0 - fast_alpha;
    const double one_minus_slow = 1.0 - slow_alpha;
    
    fast_ema[0] = prices[0];
    slow_ema[0] = prices[0];
    
    for (int i = 1; i < length; i++) {
        fast_ema[i] = fast_alpha * prices[i] + one_minus_fast * fast_ema[i - 1];
        slow_ema[i] = slow_alpha * prices[i] + one_minus_slow * slow_ema[i - 1];
    }
    
    // Calculate MACD line
    for (int i = 0; i < length; i++) {
        macd_line[i] = fast_ema[i] - slow_ema[i];
    }
    
    // Calculate signal line (EMA of MACD line)
    signal_line[0] = macd_line[0];
    const double signal_alpha = 2.0 / (signal_period + 1.0);
    const double one_minus_signal = 1.0 - signal_alpha;
    
    for (int i = 1; i < length; i++) {
        signal_line[i] = signal_alpha * macd_line[i] + one_minus_signal * signal_line[i - 1];
    }
    
    // Calculate histogram
    for (int i = 0; i < length; i++) {
        histogram[i] = macd_line[i] - signal_line[i];
    }
}

// Per-bar fitness calculation (Sharpe ratio style)
WASM_EXPORT void fitness_kernel(const double* returns, double* output, const int length, const int window) {
    if (window <= 0 || window > length) return;
    
    for (int i = window - 1; i < length; i++) {
        double sum = 0.0;
        double sum_squares = 0.0;
        
        // Calculate sum and sum of squares for window
        for (int j = 0; j < window; j++) {
            const double r = returns[i - window + j + 1];
            sum += r;
            sum_squares += r * r;
        }
        
        const double mean = sum / window;
        const double variance = (sum_squares / window) - (mean * mean);
        const double std_dev = sqrt(fmax(0.0, variance));
        
        // Sharpe ratio style fitness (mean return / std_dev)
        // Avoid division by zero
        if (std_dev > 1e-12) {
            output[i] = mean / std_dev;
        } else {
            output[i] = 0.0;
        }
    }
}