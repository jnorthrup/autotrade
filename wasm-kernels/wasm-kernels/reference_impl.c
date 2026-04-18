#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void reference_sma(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        sum += prices[i];
    }
    output[period - 1] = sum / period;
    
    for (int i = period; i < length; i++) {
        sum = sum - prices[i - period] + prices[i];
        output[i] = sum / period;
    }
}

void reference_ema(const double* prices, double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    const double alpha = 2.0 / (period + 1.0);
    
    output[0] = prices[0];
    
    for (int i = 1; i < length; i++) {
        output[i] = alpha * prices[i] + (1.0 - alpha) * output[i - 1];
    }
}

void reference_atr(const double* high, const double* low, const double* close, 
                   double* output, const int length, const int period) {
    if (period <= 0 || period > length) return;
    
    double tr = high[0] - low[0];
    output[0] = tr;
    
    for (int i = 1; i < length; i++) {
        double hl = high[i] - low[i];
        double hc = fabs(high[i] - close[i - 1]);
        double lc = fabs(low[i] - close[i - 1]);
        
        tr = hl;
        if (hc > tr) tr = hc;
        if (lc > tr) tr = lc;
        
        output[i] = (tr + (period - 1.0) * output[i - 1]) / period;
    }
}

void reference_rsi(const double* prices, double* output, const int length, const int period) {
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
    
    for (int i = period + 1; i < length; i++) {
        double diff = prices[i] - prices[i - 1];
        double gain = (diff > 0.0) ? diff : 0.0;
        double loss = (diff < 0.0) ? -diff : 0.0;
        
        avg_gain = (avg_gain * (period - 1.0) + gain) / period;
        avg_loss = (avg_loss * (period - 1.0) + loss) / period;
        
        if (avg_loss == 0.0) {
            output[i] = 100.0;
        } else {
            output[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
        }
    }
}

void generate_test_data(double* prices, double* high, double* low, double* close, int length) {
    for (int i = 0; i < length; i++) {
        prices[i] = 100.0 + 20.0 * sin(i * 0.1) + 0.5 * (rand() / (double)RAND_MAX);
        high[i] = prices[i] + 1.0 + 2.0 * (rand() / (double)RAND_MAX);
        low[i] = prices[i] - 1.0 - 2.0 * (rand() / (double)RAND_MAX);
        close[i] = prices[i] + 1.0 * (rand() / (double)RAND_MAX) - 0.5;
    }
}

int main() {
    const int length = 1000;
    const int period = 14;
    
    double* prices = malloc(length * sizeof(double));
    double* high = malloc(length * sizeof(double));
    double* low = malloc(length * sizeof(double));
    double* close = malloc(length * sizeof(double));
    
    double* ref_output = malloc(length * sizeof(double));
    
    generate_test_data(prices, high, low, close, length);
    
    printf("Reference data generated\n");
    printf("First 10 prices: ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", prices[i]);
    }
    printf("\n");
    
    reference_sma(prices, ref_output, length, period);
    printf("SMA reference (first 10): ");
    for (int i = period - 1; i < period + 10 && i < length; i++) {
        printf("%.6f ", ref_output[i]);
    }
    printf("\n");
    
    free(prices);
    free(high);
    free(low);
    free(close);
    free(ref_output);
    
    return 0;
}