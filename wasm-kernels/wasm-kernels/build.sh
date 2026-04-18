#!/bin/bash

# Build WASM module with SIMD optimization
emcc financial_kernels.c \
    -O3 \
    -msimd128 \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_sma_kernel", "_ema_kernel", "_atr_kernel", "_rsi_kernel", "_macd_kernel", "_fitness_kernel", "_vectorized_sma"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME='FinancialKernels' \
    -o financial_kernels.js

# Check for SIMD instructions
echo "Checking for SIMD instructions in WASM binary:"
wasm-objdump -d financial_kernels.wasm | grep -E "(f64x2|f64x4|v128)" | head -10

# Build native reference implementation
gcc -O3 -march=native -mfma -ffast-math reference_impl.c -o reference_impl -lm

echo "Build complete!"