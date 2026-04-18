#ifndef WASM_INDICATORS_H
#define WASM_INDICATORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==== Configuration Constants ====

#define WASM_INDICATORS_VERSION_MAJOR 1
#define WASM_INDICATORS_VERSION_MINOR 0
#define WASM_INDICATORS_VERSION_PATCH 0

#define VECTOR_WIDTH_256  4  // 256-bit vectors (4 doubles)
#define VECTOR_WIDTH_512  8  // 512-bit vectors (8 doubles)
#define CACHE_LINE_SIZE   64  // Cache line alignment

// ==== Type Definitions ====

typedef enum {
    VECTOR_256 = 256,
    VECTOR_512 = 512
} VectorWidth;

typedef enum {
    SUCCESS = 0,
    ERROR_NULL_POINTER = -1,
    ERROR_MISALIGNMENT = -2,
    ERROR_INVALID_LENGTH = -3,
    ERROR_INVALID_PERIOD = -4,
    ERROR_NAN_INF = -5,
    ERROR_BUFFER_TOO_SMALL = -6,
    ERROR_INVALID_PARAMETER = -7
} KernelError;

// Scratch buffer size calculations
#define SMA_SCRATCH_SIZE(length, period)    ((length) + (period))
#define EMA_SCRATCH_SIZE(length)            (length)  
#define ATR_SCRATCH_SIZE(length)              (3 * (length))
#define RSI_SCRATCH_SIZE(length)              (4 * (length))
#define MACD_SCRATCH_SIZE(length)             (6 * (length))

// ==== Buffer Contract Structure ====

typedef struct {
    // Input buffers - read-only
    const double* prices_in;
    const double* volumes_in;
    const double* highs_in;
    const double* lows_in;
    const double* closes_in;
    
    // Output buffers - write-only  
    double* features_out;
    double* indicators_out;
    
    // Scratch buffers - read/write
    double* temp_buffer1;
    double* temp_buffer2;
    double* temp_buffer3;
    
    // Buffer sizes - total allocated, not just used
    size_t input_length;
    size_t output_length;
    size_t temp_buffer_size;
    
    // Alignment and vectorization info
    size_t alignment;
    VectorWidth vector_width;
} BufferContract;

// ==== Ring Buffer Structure ====

typedef struct {
    double* buffer;           // Circular buffer storage
    size_t capacity;          // Total buffer capacity
    size_t head;              // Write position
    size_t tail;              // Read position
    size_t count;             // Current element count
    int    wrap_around;       // Has buffer wrapped around?
} RingBuffer;

// ==== Core Indicator Functions ====

// Simple Moving Average - vectorized
KernelError sma_vectorized(
    const double* prices,     // Input price buffer
    double* output,           // Output SMA buffer
    size_t length,            // Data length
    size_t period,            // SMA period (must be >= 1)
    double* scratch           // Scratch buffer (must be allocated)
);

// Exponential Moving Average - vectorized
KernelError ema_vectorized(
    const double* prices,     // Input price buffer
    double* output,          // Output EMA buffer
    size_t length,           // Data length
    size_t period,           // EMA period (must be >= 1)
    double* scratch          // Scratch buffer (must be allocated)
);

// Exponential Moving Average with custom alpha - vectorized
KernelError ema_vectorized_alpha(
    const double* prices,     // Input price buffer
    double* output,          // Output EMA buffer
    size_t length,           // Data length
    double alpha,            // Smoothing factor (0 < alpha <= 1)
    double* scratch          // Scratch buffer (must be allocated)
);

// Average True Range - vectorized
KernelError atr_vectorized(
    const double* highs,      // High prices buffer
    const double* lows,       // Low prices buffer
    const double* closes,     // Closing prices buffer
    double* output,           // Output ATR buffer  
    size_t length,            // Data length
    size_t period,            // ATR period (must be >= 1)
    double* scratch           // Scratch buffer (must be allocated)
);

// Relative Strength Index - vectorized
KernelError rsi_vectorized(
    const double* prices,     // Input price buffer
    double* output,           // Output RSI buffer
    size_t length,            // Data length
    size_t period,            // RSI period (typically 14, must be >= 1)
    double* scratch           // Scratch buffer (must be allocated)
);

// MACD (Moving Average Convergence Divergence) - vectorized
KernelError macd_vectorized(
    const double* prices,     // Input price buffer (prices)
    double* macd_line,        // Output MACD line buffer
    double* signal_line,      // Output signal line buffer  
    double* histogram,        // Output histogram buffer
    size_t length,            // Data length
    size_t fast_period,       // Fast EMA period (typically 12)
    size_t slow_period,       // Slow EMA period (typically 26)
    size_t signal_period,     // Signal EMA period (typically 9)
    double* scratch           // Scratch buffer (must be allocated)
);

// ==== Batch Processing Functions ====

// Process multiple indicators in sequence with shared scratch space
KernelError process_indicators_batch(
    BufferContract* contract,  // Buffer contract
    int* indicators,           // Array of indicator types
    size_t num_indicators,     // Number of indicators to process
    size_t* periods,           // Periods for each indicator
    double** outputs           // Output arrays for each indicator
);

// ==== Ring Buffer Operations ====

// Initialize ring buffer
void ring_buffer_init(
    RingBuffer* rb,
    double* buffer,
    size_t capacity
);

// Push data into ring buffer
size_t ring_buffer_push(
    RingBuffer* rb,
    const double* data,
    size_t count
);

// Pop data from ring buffer
size_t ring_buffer_pop(
    RingBuffer* rb,
    double* output,
    size_t count
);

// Get number of available elements
size_t ring_buffer_available(const RingBuffer* rb);

// Check if buffer is full
int ring_buffer_is_full(const RingBuffer* rb);

// Clear ring buffer
void ring_buffer_clear(RingBuffer* rb);

// ==== Utility Functions ====

// Memory allocation with alignment
void* wasm_aligned_alloc(size_t size, size_t alignment);

// Free aligned memory  
void wasm_aligned_free(void* ptr);

// Get kernel version
void get_kernel_version(int* major, int* minor, int* patch);

// Get error string
const char* kernel_error_string(KernelError error);

// Validate buffer alignment
int validate_buffer_alignment(const void* ptr, size_t alignment);

// Validate IEEE-754 double precision range
int validate_f64_range(double value);

// Check for NaN/Inf values
KernelError ensure_no_nan_inf(const double* buffer, size_t length);

// Get optimal vector width for current CPU
VectorWidth get_optimal_vector_width(void);

// Set CPU-specific optimizations
void set_vector_width(VectorWidth width);

// ==== Memory Management ====

typedef struct {
    void* base_ptr;           // Base allocation pointer
    size_t total_size;        // Total allocated size
    size_t used_size;         // Currently used size
    size_t alignment;         // Required alignment
} MemoryPool;

// Initialize memory pool from WASM linear memory
MemoryPool* memory_pool_init(
    uint8_t* wasm_memory,
    size_t offset, 
    size_t size,
    size_t alignment
);

// Allocate from memory pool
void* memory_pool_alloc(MemoryPool* pool, size_t size);

// Reset memory pool (free all allocations)
void memory_pool_reset(MemoryPool* pool);

#ifdef __cplusplus
}
#endif

#endif // WASM_INDICATORS_H