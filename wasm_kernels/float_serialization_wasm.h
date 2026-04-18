/**
 * High-Precision Float Deserialization for WASM
 * 
 * C++ implementation for deserializing 64-bit IEEE-754 floats
 * from JavaScript with full precision preservation.
 * 
 * Features:
 * - Full IEEE-754 double precision support
 * - Endianness handling
 * - Checksum validation
 * - Optimized for WASM compilation
 * - Zero-copy operations where possible
 */

#ifndef FLOAT_SERIALIZATION_WASM_H
#define FLOAT_SERIALIZATION_WASM_H

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>

namespace autotrade {
namespace serialization {

/**
 * Float serialization/deserialization utilities for WASM
 */
class FloatDeserializer {
private:
    uint32_t fnv1a_checksum_32(const uint8_t* data, size_t length) const {
        const uint32_t FNV_32_PRIME = 0x01000193;
        const uint32_t FNV_32_OFFSET = 0x811c9dc5;
        
        uint32_t hash = FNV_32_OFFSET;
        
        for (size_t i = 0; i < length; ++i) {
            hash ^= data[i];
            hash *= FNV_32_PRIME;
            hash &= 0xFFFFFFFF;
        }
        
        return hash;
    }
    
public:
    /**
     * Deserialize a single float from binary data with checksum validation
     * Format: [8 bytes float][4 bytes checksum]
     * 
     * @param data Pointer to serialized data
     * @param length Length of data buffer (must be >= 12)
     * @param offset Start offset in buffer
     * @return Deserialized double value
     */
    double deserialize_float(const uint8_t* data, size_t length, size_t offset = 0) const {
        if (offset + 12 > length) {
            throw std::runtime_error("Buffer too small for float deserialization");
        }
        
        // Extract float bytes (little-endian)
        uint8_t float_bytes[8];
        std::memcpy(float_bytes, data + offset, 8);
        
        // Convert to double
        double value;
        std::memcpy(&value, float_bytes, sizeof(double));
        
        // Calculate checksum
        uint32_t calculated_checksum = fnv1a_checksum_32(float_bytes, 8);
        
        // Extract and verify checksum (little-endian)
        uint32_t expected_checksum = *reinterpret_cast<const uint32_t*>(data + offset + 8);
        
        if (expected_checksum != calculated_checksum) {
            throw std::runtime_error("Checksum mismatch");
        }
        
        return value;
    }
    
    /**
     * Deserialize array of floats
     * 
     * @param data Pointer to serialized data
     * @param count Number of floats to deserialize
     * @param length Length of data buffer
     * @param offset Start offset in buffer
     * @return Vector of deserialized doubles
     */
    std::vector<double> deserialize_float_array(
        const uint8_t* data, 
        size_t count, 
        size_t length, 
        size_t offset = 0
    ) const {
        if (offset + count * 12 > length) {
            throw std::runtime_error("Buffer too small for array deserialization");
        }
        
        std::vector<double> result;
        result.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            result.push_back(deserialize_float(data, length, offset + i * 12));
        }
        
        return result;
    }
    
    /**
     * Extract checksum from serialized data
     */
    uint32_t extract_checksum(const uint8_t* data, size_t offset = 0) const {
        return *reinterpret_cast<const uint32_t*>(data + offset + 8);
    }
    
    /**
     * Calculate checksum for string of bytes
     */
    uint32_t calculate_checksum(const uint8_t* data, size_t length) const {
        return fnv1a_checksum_32(data, length);
    }
    
    /**
     * Bit-exact comparison of two doubles
     */
    bool bit_exact_compare(double a, double b) const {
        uint64_t a_bits, b_bits;
        std::memcpy(&a_bits, &a, sizeof(double));
        std::memcpy(&b_bits, &b, sizeof(double));
        return a_bits == b_bits;
    }
    
    /**
     * Get raw bytes of double value
     */
    std::vector<uint8_t> get_double_bytes(double value) const {
        std::vector<uint8_t> result(8);
        std::memcpy(result.data(), &value, sizeof(double));
        return result;
    }
    
    /**
     * Detect host endianness
     */
    bool is_little_endian() const {
        uint16_t test = 0x1234;
        uint8_t byte1 = *reinterpret_cast<uint8_t*>(&test);
        return byte1 == 0x34;
    }
};

/**
 * WASM-compatible C interface
 * These functions can be called from JavaScript/WebAssembly
 */

extern "C" {
    /**
     * Deserialize single float (WASM interface)
     * 
     * @param data Pointer to serialized data in WASM memory
     * @param length Length of data buffer
     * @param offset Start offset
     * @return Deserialized double value
     */
    
    double wasm_deserialize_float(const uint8_t* data, size_t length, size_t offset) {
        static FloatDeserializer deserializer;
        try {
            return deserializer.deserialize_float(data, length, offset);
        } catch (const std::exception& e) {
            // Return NaN on error for WASM compatibility
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    
    /**
     * Deserialize array of floats (WASM interface)
     * 
     * @param data Pointer to serialized data
     * @param count Number of floats
     * @param length Length of data buffer
     * @param offset Start offset
     * @param result_buffer Pointer to result buffer (preallocated by JS)
     */
    void wasm_deserialize_float_array(
        const uint8_t* data,
        size_t count,
        size_t length,
        size_t offset,
        double* result_buffer
    ) {
        static FloatDeserializer deserializer;
        try {
            auto result = deserializer.deserialize_float_array(data, count, length, offset);
            std::memcpy(result_buffer, result.data(), count * sizeof(double));
        } catch (const std::exception& e) {
            // Fill with NaN on error
            for (size_t i = 0; i < count; ++i) {
                result_buffer[i] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    
    /**
     * Verify checksum for serialized data (WASM interface)
     */
    bool wasm_verify_float_checksum(const uint8_t* data, size_t length, size_t offset) {
        static FloatDeserializer deserializer;
        try {
            deserializer.deserialize_float(data, length, offset);
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    /**
     * Get host endianness (WASM interface)
     */
    bool wasm_is_little_endian() {
        static FloatDeserializer deserializer;
        return deserializer.is_little_endian();
    }
    
    /**
     * Bit-exact comparison (WASM interface)
     */
    bool wasm_bit_exact_compare(double a, double b) {
        static FloatDeserializer deserializer;
        return deserializer.bit_exact_compare(a, b);
    }
}

/**
 * Performance optimization utilities
 */
class FloatDeserializerOptimized : public FloatDeserializer {
public:
    /**
     * Batch deserialize multiple floats with memory pre-allocation
     * Optimized for high-throughput scenarios
     * 
     * @param data Pointer to serialized data
     * @param count Number of floats
     * @param length Length of data buffer
     * @param offset Start offset
     * @return Vector of deserialized doubles
     */
    std::vector<double> deserialize_float_array_batch(
        const uint8_t* data, 
        size_t count, 
        size_t length, 
        size_t offset = 0
    ) {
        std::vector<double> result;
        result.reserve(count);
        
        size_t current_offset = offset;
        
        for (size_t i = 0; i < count; ++i) {
            if (current_offset + 12 > length) {
                result.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            
            // Fast path: minimal error checking for batch processing
            uint8_t float_bytes[8];
            std::memcpy(float_bytes, data + current_offset, 8);
            
            // Simple checksum verification
            uint32_t calculated_checksum = fnv1a_checksum_32(float_bytes, 8);
            uint32_t expected_checksum = *reinterpret_cast<const uint32_t*>(data + current_offset + 8);
            
            double value;
            std::memcpy(&value, float_bytes, sizeof(double));
            
            // Only reject if checksum mismatch AND not naturally matching bytes
            if (expected_checksum != calculated_checksum) {
                value = std::numeric_limits<double>::quiet_NaN();
            }
            
            result.push_back(value);
            current_offset += 12;
        }
        
        return result;
    }
    
    /**
     * Get performance metrics
     */
    struct PerformanceMetrics {
        size_t successful_deserializations = 0;
        size_t checksum_failures = 0;
        double processing_rate_bytes_per_second = 0.0;
        double throughput_values_per_second = 0.0;
    };
    
    PerformanceMetrics get_last_metrics() const {
        return PerformanceMetrics();
    }
};

} // namespace serialization
} // namespace autotrade

#endif // FLOAT_SERIALIZATION_WASM_H