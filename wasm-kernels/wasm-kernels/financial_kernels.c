#include <emscripten/emscripten.h>
#include <math.h>
#include <stdint.h>

#define VECTOR_SIZE 4

typedef double v2d __attribute__((vector_size(16)));
typedef double v4d __attribute__((vector_size(32)));

EMSCRIPTEN_KEEPALIVE
void sma_kernel(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        sum += prices[i];
    }
    output[period - 1] = sum / period;
    
    const double inv_period = 1.0 / period;
    for (int i = period; i < length; i++) {
        sum = sum - prices[i - period] + prices[i];
        output[i] = sum * inv_period;
    }
}

EMSCRIPTEN_KEEPALIVE
void ema_kernel(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    const double alpha = 2.0 / (period + 1.0);
    const double one_minus_alpha = 1.0 - alpha;
    
    output[0] = prices[0];
    
    for (int i = 1; i < length; i++) {
        output[i] = alpha * prices[i] + one_minus_alpha * output[i - 1];
    }
}

EMSCRIPTEN_KEEPALIVE
void atr_kernel(const double* high, const double* low, const double* close, 
                 double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    double tr = high[0] - low[0];
    output[0] = tr;
    
    const double alpha = 1.0 / period;
    const double one_minus_alpha = 1.0 - alpha;
    
    for (int i = 1; i < length; i++) {
        double hl = high[i] - low[i];
        double hc = fabs(high[i] - close[i - 1]);
        double lc = fabs(low[i] - close[i - 1]);
        
        tr = hl;
        if (hc > tr) tr = hc;
        if (lc > tr) tr = lc;
        
        output[i] = alpha * tr + one_minus_alpha * output[i - 1];
    }
}

EMSCRIPTEN_KEEPALIVE
void rsi_kernel(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period >= length) return;
    
    double gain_sum = 0.0;
    double loss_sum = 0.0;
    
    for (int i = 1; i <= period; i++) {
        double diff = prices[i] - prices[i - 1];
        if (diff > 0.0) {
            gain_sum += diff;
        } else {
            loss_sum += -diff;
        }
    }
    
    double avg_gain = gain_sum / period;
    double avg_loss = loss_sum / period;
    
    if (avg_loss == 0.0) {
        output[period] = 100.0;
    } else {
        output[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
    }
    
    const double alpha = 1.0 / period;
    const double one_minus_alpha = 1.0 - alpha;
    
    for (int i = period + 1; i < length; i++) {
        double diff = prices[i] - prices[i - 1];
        double gain = (diff > 0.0) ? diff : 0.0;
        double loss = (diff < 0.0) ? -diff : 0.0;
        
        avg_gain = alpha * gain + one_minus_alpha * avg_gain;
        avg_loss = alpha * loss + one_minus_alpha * avg_loss;
        
        if (avg_loss == 0.0) {
            output[i] = 100.0;
        } else {
            output[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
        }
    }
}

EMSCRIPTEN_KEEPALIVE
void macd_kernel(const double* prices, double* macd_line, double* signal_line, 
                 double* histogram, const int length, const int fast_period, const int slow_period, const int signal_period) {
    if (fast_period <= 0 || slow_period <= fast_period || signal_period <= 0) return;
    
    double fast_alpha = 2.0 / (fast_period + 1.0);
    double slow_alpha = 2.0 / (slow_period + 1.0);
    double signal_alpha = 2.0 / (signal_period + 1.0);
    
    double fast_ema = prices[0];
    double slow_ema = prices[0];
    
    for (int i = 1; i < length; i++) {
        fast_ema = fast_alpha * prices[i] + (1.0 - fast_alpha) * fast_ema;
        slow_ema = slow_alpha * prices[i] + (1.0 - slow_alpha) * slow_ema;
        
        macd_line[i] = fast_ema - slow_ema;
    }
    
    signal_line[0] = macd_line[0];
    for (int i = 1; i < length; i++) {
        signal_line[i] = signal_alpha * macd_line[i] + (1.0 - signal_alpha) * signal_line[i - 1];
        histogram[i] = macd_line[i] - signal_line[i];
    }
}

EMSCRIPTEN_KEEPALIVE
double fitness_kernel(const double* prices, const double* signals, const int length, 
                      const double transaction_cost, const double risk_free_rate) {
    if (length <= 1) return 0.0;
    
    double total_return = 0.0;
    double position = 0.0;
    double max_drawdown = 0.0;
    double peak = 0.0;
    double current_value = 1.0;
    
    for (int i = 1; i < length; i++) {
        double signal = signals[i];
        double price_change = (prices[i] - prices[i - 1]) / prices[i - 1];
        
        if (signal > 0.0 && position <= 0.0) {
            position = 1.0;
            current_value *= (1.0 - transaction_cost);
        } else if (signal < 0.0 && position >= 0.0) {
            position = -1.0;
            current_value *= (1.0 - transaction_cost);
        }
        
        current_value *= (1.0 + position * price_change);
        
        if (current_value > peak) {
            peak = current_value;
        }
        
        double drawdown = (peak - current_value) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }
    
    total_return = current_value - 1.0;
    double sharpe_ratio = (total_return - risk_free_rate) / sqrt(fabs(total_return) + 1e-15);
    
    return sharpe_ratio - max_drawdown;
}

EMSCRIPTEN_KEEPALIVE
void vectorized_sma(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period > length || period < 4) {
        sma_kernel(prices, output, length, period);
        return;
    }
    
    double sum = 0.0;
    int i;
    
    for (i = 0; i <= period - 4; i += 4) {
        v4d vec = *(v4d*)&prices[i];
        sum += vec[0] + vec[1] + vec[2] + vec[3];
    }
    
    for (; i < period; i++) {
        sum += prices[i];
    }
    
    output[period - 1] = sum / period;
    
    const double inv_period = 1.0 / period;
    const v4d v_inv_period = {inv_period, inv_period, inv_period, inv_period};
    
    for (i = period; i <= length - 4; i += 4) {
        v4d old_prices = *(v4d*)&prices[i - period];
        v4d new_prices = *(v4d*)&prices[i];
        
        sum = sum - (old_prices[0] + old_prices[1] + old_prices[2] + old_prices[3]) +
                   (new_prices[0] + new_prices[1] + new_prices[2] + new_prices[3]);
        
        v4d result = {sum * inv_period, sum * inv_period, sum * inv_period, sum * inv_period};
        *(v4d*)&output[i] = result;
    }
    
    for (; i < length; i++) {
        sum = sum - prices[i - period] + prices[i];
        output[i] = sum * inv_period;
    }
}