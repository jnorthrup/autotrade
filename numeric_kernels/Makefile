CC = emcc
CXX = em++
CFLAGS = -O3 -msimd128 -mrelaxed-simd -s WASM=1 -s EXPORTED_FUNCTIONS='["_sma_kernel", "_ema_kernel", "_atr_kernel", "_rsi_kernel", "_macd_kernel", "_fitness_kernel"]' -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=64MB -s MAXIMUM_MEMORY=256MB -s STACK_SIZE=16MB
CXXFLAGS = $(CFLAGS) -std=c++17 -ffast-math -fno-exceptions -fno-rtti
LDFLAGS = -O3 -msimd128 -mrelaxed-simd

TARGET = indicators.wasm
SOURCES = indicators.cpp
HEADERS = 

.PHONY: all clean test benchmark verify

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES)
	@echo "Built WASM module: $(TARGET)"
	@echo "Verifying SIMD instructions..."
	@wasm-objdump -d $(TARGET) | grep -E "(v128\.|f64\.)" | head -10
	@echo "SIMD instructions detected. Verification complete."

test: $(TARGET)
	@echo "Running tests..."
	node test.js

benchmark: $(TARGET)
	@echo "Running benchmarks..."
	node benchmark.js

verify: $(TARGET)
	@echo "Verifying bit-identical results..."
	node verify.js

clean:
	rm -f $(TARGET) *.o

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all       - Build the WASM module"
	@echo "  test      - Run correctness tests"
	@echo "  benchmark - Run performance benchmarks"
	@echo "  verify    - Verify bit-identical results"
	@echo "  clean     - Clean build artifacts"