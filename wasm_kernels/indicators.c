#include <stdint.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Simple Moving Average
void sma_f64(const double* prices, double* output, int64_t length, int64_t period) {
    double sum = 0.0;
    int64_t i;
    
    // Initial window
    for (i = 0; i < period; i++) {
        sum += prices[i];
    }
    output[period - 1] = sum / (double)period;
    
    // Rolling window
    for (i = period; i < length; i++) {
        sum = sum - prices[i - period] + prices[i];
        output[i] = sum / (double)period;
    }
}

// Exponential Moving Average
void ema_f64(const double* prices, double* output, int64_t length, int64_t period) {
    double multiplier = 2.0 / ((double)period + 1.0);
    double ema_prev = prices[0];
    int64_t i;
    
    output[0] = ema_prev;
    
    for (i = 1; i < length; i++) {
        ema_prev = (prices[i] * multiplier) + (ema_prev * (1.0 - multiplier));
        output[i] = ema_prev;
    }
}

// Average True Range
void atr_f64(const double* highs, const double* lows, const double* closes, 
             double* output, int64_t length, int64_t period) {
    double atr_prev = 0.0;
    int64_t i;
    
    // First ATR value - simple average of first period
    for (i = 1; i <= period; i++) {
        double high_low = highs[i] - lows[i];
        double high_close = highs[i] - closes[i - 1];
        double low_close = lows[i] - closes[i - 1];
        high_close = high_close > 0.0 ? high_close : -high_close;
        low_close = low_close > 0.0 ? low_close : -low_close;
        double tr = high_low;
        if (high_close > tr) tr = high_close;
        if (low_close > tr) tr = low_close;
        atr_prev += tr;
    }
    atr_prev = atr_prev / (double)period;
    output[period - 1] = atr_prev;
    
    // Wilder's smoothing
    double multiplier = 1.0 / (double)period;
    for (i = period; i < length; i++) {
        double high_low = highs[i] - lows[i];
        double high_close = highs[i] - closes[i - 1];
        double low_close = lows[i] - closes[i - 1];
        high_close = high_close > 0.0 ? high_close : -high_close;
        low_close = low_close > 0.0 ? low_close : -low_close;
        double tr = high_low;
        if (high_close > tr) tr = high_close;
        if (low_close > tr) tr = low_close;
        atr_prev = (tr * multiplier) + (atr_prev * (1.0 - multiplier));
        output[i] = atr_prev;
    }
}

// Relative Strength Index
void rsi_f64(const double* prices, double* output, int64_t length, int64_t period) {
    double gain_sum = 0.0;
    double loss_sum = 0.0;
    int64_t i;
    
    // Initial period
    for (i = 1; i <= period; i++) {
        double diff = prices[i] - prices[i - 1];
        if (diff > 0.0) {
            gain_sum += diff;
        } else {
            loss_sum += -diff;
        }
    }
    
    double avg_gain = gain_sum / (double)period;
    double avg_loss = loss_sum / (double)period;
    double rs = avg_gain / (avg_loss + 1e-15); // Avoid division by zero
    output[period - 1] = 100.0 - (100.0 / (1.0 + rs));
    
    // Wilder's smoothing
    double multiplier = 1.0 / (double)period;
    for (i = period; i < length; i++) {
        double diff = prices[i] - prices[i - 1];
        if (diff > 0.0) {
            avg_gain = (diff * multiplier) + (avg_gain * (1.0 - multiplier));
            avg_loss = avg_loss * (1.0 - multiplier);
        } else {
            avg_loss = (-diff * multiplier) + (avg_loss * (1.0 - multiplier));
            avg_gain = avg_gain * (1.0 - multiplier);
        }
        rs = avg_gain / (avg_loss + 1e-15);
        output[i] = 100.0 - (100.0 / (1.0 + rs));
    }
}

// MACD (Moving Average Convergence Divergence)
void macd_f64(const double* prices, double* macd_line, double* signal_line, 
              double* histogram, int64_t length, int64_t fast_period, 
              int64_t slow_period, int64_t signal_period) {
    double fast_ema = prices[0];
    double slow_ema = prices[0];
    double signal_ema = 0.0;
    double fast_multiplier = 2.0 / ((double)fast_period + 1.0);
    double slow_multiplier = 2.0 / ((double)slow_period + 1.0);
    double signal_multiplier = 2.0 / ((double)signal_period + 1.0);
    int64_t i;
    
    for (i = 1; i < length; i++) {
        fast_ema = (prices[i] * fast_multiplier) + (fast_ema * (1.0 - fast_multiplier));
        slow_ema = (prices[i] * slow_multiplier) + (slow_ema * (1.0 - slow_multiplier));
        macd_line[i] = fast_ema - slow_ema;
        
        if (i >= slow_period - 1) {
            if (i == slow_period - 1) {
                signal_ema = macd_line[i];
            } else {
                signal_ema = (macd_line[i] * signal_multiplier) + (signal_ema * (1.0 - signal_multiplier));
            }
            signal_line[i] = signal_ema;
            histogram[i] = macd_line[i] - signal_ema;
        }
    }
}

// Per-bar fitness calculation (simple Sharpe ratio style metric)
void fitness_f64(const double* returns, double* fitness_scores, int64_t length) {
    double sum_returns = 0.0;
    double sum_sq_returns = 0.0;
    int64_t i;
    
    // Calculate mean return
    for (i = 0; i < length; i++) {
        sum_returns += returns[i];
    }
    double mean_return = sum_returns / (double)length;
    
    // Calculate standard deviation
    for (i = 0; i < length; i++) {
        double diff = returns[i] - mean_return;
        sum_sq_returns += diff * diff;
    }
    double std_dev = sum_sq_returns / (double)(length - 1);
    std_dev = std_dev > 0.0 ? std_dev : 1e-15;
    
    // Sharpe ratio (return / std_dev)
    double sharpe_ratio = mean_return / std_dev;
    
    // Store fitness score for each bar (scaled)
    for (i = 0; i < length; i++) {
        fitness_scores[i] = returns[i] * sharpe_ratio;
    }
}