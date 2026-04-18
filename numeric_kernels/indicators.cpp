#include <emscripten/emscripten.h>
#include <cmath>
#include <algorithm>
#include <cstring>

extern "C" {

// SIMD optimization macros
#define VECTOR_WIDTH 2
#define ALIGNMENT 32

// Helper function for aligned memory allocation
static double* aligned_alloc(size_t count) {
    return (double*)aligned_alloc(ALIGNMENT, count * sizeof(double));
}

// Simple Moving Average (SMA) - tight monomorphic loop
EMSCRIPTEN_KEEPALIVE
void sma_kernel(const double* prices, double* output, int length, int period) {
    if (period <= 0 || length <= 0 || period > length) return;
    
    const double inv_period = 1.0 / period;
    
    // Initialize first value
    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        sum += prices[i];
    }
    output[period - 1] = sum * inv_period;
    
    // Rolling sum for remaining values - monomorphic loop
    for (int i = period; i < length; i++) {
        sum = sum - prices[i - period] + prices[i];
        output[i] = sum * inv_period;
    }
}

// Exponential Moving Average (EMA) - tight monomorphic loop
EMSCRIPTEN_KEEPALIVE
void ema_kernel(const double* prices, double* output, int length, int period) {
    if (period <= 0 || length <= 0) return;
    
    const double alpha = 2.0 / (period + 1.0);
    const double one_minus_alpha = 1.0 - alpha;
    
    // Initialize with SMA
    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        sum += prices[i];
    }
    output[period - 1] = sum / period;
    
    // EMA calculation - monomorphic loop
    for (int i = period; i < length; i++) {
        output[i] = alpha * prices[i] + one_minus_alpha * output[i - 1];
    }
}

// Average True Range (ATR) - tight monomorphic loop
EMSCRIPTEN_KEEPALIVE
void atr_kernel(const double* highs, const double* lows, const double* closes, 
                double* output, int length, int period) {
    if (period <= 0 || length <= 0 || period > length) return;
    
    // Calculate True Range values
    double* tr_values = aligned_alloc(length);
    
    // First TR value
    tr_values[0] = highs[0] - lows[0];
    
    // Remaining TR values - monomorphic loop
    for (int i = 1; i < length; i++) {
        const double hl = highs[i] - lows[i];
        const double hc = std::abs(highs[i] - closes[i - 1]);
        const double lc = std::abs(lows[i] - closes[i - 1]);
        
        // Vectorizable max operations
        double max1 = (hl > hc) ? hl : hc;
        tr_values[i] = (max1 > lc) ? max1 : lc;
    }
    
    // Calculate ATR using Wilder's smoothing
    const double alpha = 1.0 / period;
    const double one_minus_alpha = 1.0 - alpha;
    
    // Initialize with average of first period TR values
    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        sum += tr_values[i];
    }
    output[period - 1] = sum / period;
    
    // ATR smoothing - monomorphic loop
    for (int i = period; i < length; i++) {
        output[i] = alpha * tr_values[i] + one_minus_alpha * output[i - 1];
    }
    
    free(tr_values);
}

// Relative Strength Index (RSI) - tight monomorphic loop
EMSCRIPTEN_KEEPALIVE
void rsi_kernel(const double* prices, double* output, int length, int period) {
    if (period <= 0 || length <= 0 || period >= length) return;
    
    const double inv_period = 1.0 / period;
    
    // Calculate price changes
    double* gains = aligned_alloc(length);
    double* losses = aligned_alloc(length);
    
    // First price change
    const double first_change = prices[1] - prices[0];
    gains[1] = (first_change > 0) ? first_change : 0.0;
    losses[1] = (first_change < 0) ? -first_change : 0.0;
    
    // Remaining price changes - monomorphic loop
    for (int i = 2; i < length; i++) {
        const double change = prices[i] - prices[i - 1];
        gains[i] = (change > 0) ? change : 0.0;
        losses[i] = (change < 0) ? -change : 0.0;
    }
    
    // Calculate initial average gain and loss
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    for (int i = 1; i <= period; i++) {
        avg_gain += gains[i];
        avg_loss += losses[i];
    }
    avg_gain *= inv_period;
    avg_loss *= inv_period;
    
    // Calculate initial RSI
    if (avg_loss == 0.0) {
        output[period] = 100.0;
    } else {
        const double rs = avg_gain / avg_loss;
        output[period] = 100.0 - (100.0 / (1.0 + rs));
    }
    
    // Smooth and calculate remaining RSI values - monomorphic loop
    for (int i = period + 1; i < length; i++) {
        avg_gain = (avg_gain * (period - 1) + gains[i]) * inv_period;
        avg_loss = (avg_loss * (period - 1) + losses[i]) * inv_period;
        
        if (avg_loss == 0.0) {
            output[i] = 100.0;
        } else {
            const double rs = avg_gain / avg_loss;
            output[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    free(gains);
    free(losses);
}

// MACD (Moving Average Convergence Divergence) - tight monomorphic loop
EMSCRIPTEN_KEEPALIVE
void macd_kernel(const double* prices, double* macd_line, double* signal_line, 
                 double* histogram, int length, int fast_period, int slow_period, int signal_period) {
    if (fast_period <= 0 || slow_period <= 0 || signal_period <= 0 || 
        fast_period >= slow_period || length <= 0) return;
    
    // Calculate fast EMA
    double* fast_ema = aligned_alloc(length);
    const double fast_alpha = 2.0 / (fast_period + 1.0);
    const double fast_one_minus_alpha = 1.0 - fast_alpha;
    
    double sum = 0.0;
    for (int i = 0; i < fast_period; i++) {
        sum += prices[i];
    }
    fast_ema[fast_period - 1] = sum / fast_period;
    
    for (int i = fast_period; i < length; i++) {
        fast_ema[i] = fast_alpha * prices[i] + fast_one_minus_alpha * fast_ema[i - 1];
    }
    
    // Calculate slow EMA
    double* slow_ema = aligned_alloc(length);
    const double slow_alpha = 2.0 / (slow_period + 1.0);
    const double slow_one_minus_alpha = 1.0 - slow_alpha;
    
    sum = 0.0;
    for (int i = 0; i < slow_period; i++) {
        sum += prices[i];
    }
    slow_ema[slow_period - 1] = sum / slow_period;
    
    for (int i = slow_period; i < length; i++) {
        slow_ema[i] = slow_alpha * prices[i] + slow_one_minus_alpha * slow_ema[i - 1];
    }
    
    // Calculate MACD line - monomorphic loop
    int start_idx = (slow_period > fast_period) ? slow_period - 1 : fast_period - 1;
    for (int i = 0; i < length; i++) {
        if (i < start_idx) {
            macd_line[i] = 0.0;
        } else {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }
    
    // Calculate signal line (EMA of MACD)
    const double signal_alpha = 2.0 / (signal_period + 1.0);
    const double signal_one_minus_alpha = 1.0 - signal_alpha;
    
    sum = 0.0;
    int signal_start = start_idx;
    for (int i = signal_start; i < signal_start + signal_period && i < length; i++) {
        sum += macd_line[i];
    }
    
    if (signal_start + signal_period <= length) {
        signal_line[signal_start + signal_period - 1] = sum / signal_period;
        
        // Signal line smoothing - monomorphic loop
        for (int i = signal_start + signal_period; i < length; i++) {
            signal_line[i] = signal_alpha * macd_line[i] + signal_one_minus_alpha * signal_line[i - 1];
        }
        
        // Calculate histogram - monomorphic loop
        for (int i = signal_start + signal_period - 1; i < length; i++) {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }
    
    free(fast_ema);
    free(slow_ema);
}

// Per-bar fitness calculation - tight monomorphic loop
EMSCRIPTEN_KEEPALIVE
void fitness_kernel(const double* prices, const double* signals, double* fitness, 
                    int length, double risk_free_rate, double fitness_window) {
    if (length <= 0 || fitness_window <= 0) return;
    
    const int window_size = (int)fitness_window;
    if (window_size <= 0 || window_size > length) return;
    
    // Pre-calculate returns
    double* returns = aligned_alloc(length);
    for (int i = 1; i < length; i++) {
        returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
    }
    
    // Calculate fitness for each bar - monomorphic loop
    for (int i = window_size; i < length; i++) {
        double total_return = 0.0;
        double signal_following_return = 0.0;
        double squared_returns = 0.0;
        int signal_trades = 0;
        
        // Calculate statistics over window - vectorizable loop
        for (int j = i - window_size + 1; j <= i; j++) {
            const double ret = returns[j];
            total_return += ret;
            squared_returns += ret * ret;
            
            // Signal following logic
            if (signals[j] > 0.0) {
                signal_following_return += ret;
                signal_trades++;
            }
        }
        
        // Calculate fitness metrics
        const double mean_return = total_return / window_size;
        const double variance = (squared_returns - (total_return * total_return) / window_size) / (window_size - 1);
        const double volatility = std::sqrt(std::max(0.0, variance));
        
        // Sharpe ratio
        double sharpe_ratio = 0.0;
        if (volatility > 0.0) {
            sharpe_ratio = (mean_return - risk_free_rate / 252.0) / (volatility * std::sqrt(252.0));
        }
        
        // Signal effectiveness
        double signal_effectiveness = 0.0;
        if (signal_trades > 0) {
            signal_effectiveness = signal_following_return / signal_trades;
        }
        
        // Combined fitness score
        fitness[i] = 0.6 * sharpe_ratio + 0.4 * signal_effectiveness;
    }
    
    free(returns);
}

}