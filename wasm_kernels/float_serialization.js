/**
 * High-Precision Float Serialization for JS-WASM Communication
 * 
 * Implements binary serialization/deserialization for 64-bit IEEE-754 floats
 * with full precision preservation, including subnormals, NaN payloads, and edge cases.
 * 
 * Features:
 * - Roundtrip fidelity for all IEEE-754 double values
 * - Endianness handling
 * - Checksum validation
 * - High performance for real-time processing
 */

class FloatSerializer {
  constructor() {
    // Determine host endianness
    this.isLittleEndian = this._detectEndianness();
    this.buffer = new ArrayBuffer(8);
    this.view = new DataView(this.buffer);
  }

  /**
   * Detect system endianness using a known bit pattern
   */
  _detectEndianness() {
    const test = new Uint16Array([0x1234]);
    const bytes = new Uint8Array(test.buffer);
    return bytes[0] === 0x34;
  }

  /**
   * Serialize a single 64-bit float with checksum
   * Format: [8 bytes float][4 bytes checksum]
   */
  serializeFloat(value) {
    // Set float in buffer
    this.view.setFloat64(0, value);
    
    // Calculate checksum
    const checksum = this._calculateChecksum();
    
    // Create result buffer: 8 bytes + 4 bytes checksum
    const result = new ArrayBuffer(12);
    const resultView = new DataView(result);
    
    // Copy float bytes
    const bytes = new Uint8Array(this.buffer);
    const resultBytes = new Uint8Array(result);
    resultBytes.set(bytes, 0);
    
    // Add checksum (little-endian for consistency)
    resultView.setUint32(8, checksum, true);
    
    return result;
  }

  /**
   * Serialize array of floats
   */
  serializeFloatArray(values) {
    const totalBytes = values.length * 12; // 8 bytes + 4 bytes checksum per float
    const result = new ArrayBuffer(totalBytes);
    const resultView = new DataView(result);
    
    for (let i = 0; i < values.length; i++) {
      const offset = i * 12;
      
      // Serialize individual float
      this.view.setFloat64(0, values[i]);
      const bytes = new Uint8Array(this.buffer);
      const checksum = this._calculateChecksum();
      
      // Copy to result buffer
      const resultBytes = new Uint8Array(result, offset, 12);
      resultBytes.set(bytes, 0);
      new DataView(result).setUint32(offset + 8, checksum, true);
    }
    
    return result;
  }

  /**
   * Deserialize float with checksum validation
   */
  deserializeFloat(buffer, offset = 0) {
    if (buffer.byteLength < offset + 12) {
      throw new Error('Buffer too small for float deserialization');
    }
    
    const view = new DataView(buffer);
    
    // Extract float bytes
    const floatBytes = new ArrayBuffer(8);
    const floatBytesView = new Uint8Array(floatBytes);
    const sourceBytes = new Uint8Array(buffer, offset, 8);
    floatBytesView.set(sourceBytes);
    
    // Extract checksum
    const expectedChecksum = view.getUint32(offset + 8, true);
    
    // Verify checksum
    this.buffer = floatBytes;
    this.view = new DataView(this.buffer);
    const calculatedChecksum = this._calculateChecksum();
    
    if (expectedChecksum !== calculatedChecksum) {
      throw new Error(`Checksum mismatch: expected ${expectedChecksum}, got ${calculatedChecksum}`);
    }
    
    // Return the float value
    return new DataView(floatBytes).getFloat64(0);
  }

  /**
   * Deserialize array of floats
   */
  deserializeFloatArray(buffer, count) {
    const result = new Array(count);
    
    for (let i = 0; i < count; i++) {
      const offset = i * 12;
      result[i] = this.deserializeFloat(buffer, offset);
    }
    
    return result;
  }

  /**
   * Calculate checksum using 32-bit FNV-1a hash
   */
  _calculateChecksum() {
    const FNV_32_PRIME = 0x01000193;
    const FNV_32_OFFSET = 0x811c9dc5;
    
    let hash = FNV_32_OFFSET;
    const bytes = new Uint8Array(this.buffer);
    
    for (let i = 0; i < bytes.length; i++) {
      hash ^= bytes[i];
      hash *= FNV_32_PRIME;
      hash >>>= 0; // Ensure 32-bit unsigned
    }
    
    return hash >>> 0;
  }

  /**
   * Get raw bytes of float value for bit-exact comparison
   */
  getFloatBytes(value) {
    this.view.setFloat64(0, value);
    return new Uint8Array(this.buffer.slice());
  }

  /**
   * Compare two float values bit-exact
   */
  bitExactCompare(value1, value2) {
    const bytes1 = this.getFloatBytes(value1);
    const bytes2 = this.getFloatBytes(value2);
    
    return Array.from(bytes1).every((byte, i) => byte === bytes2[i]);
  }

  /**
   * Get system endianness
   */
  getEndianness() {
    return this.isLittleEndian ? 'little' : 'big';
  }
}

/**
 * WASM-compatible serialization functions
 * These can be called from WASM for native performance
 */
const WASMFloatSerializer = {
  /**
   * Serializes float array to WASM memory
   * Returns pointer to serialized data
   */
  serializeToWASM: function(wasmMemory, values, allocator) {
    const serializer = new FloatSerializer();
    const buffer = serializer.serializeFloatArray(values);
    const size = buffer.byteLength;
    const ptr = allocator(size);
    
    if (ptr === 0) {
      throw new Error('Memory allocation failed');
    }
    
    const wasmMemoryView = new Uint8Array(wasmMemory.buffer, ptr, size);
    const sourceBytes = new Uint8Array(buffer);
    wasmMemoryView.set(sourceBytes);
    
    return { ptr, size };
  },

  /**
   * Deserializes from WASM memory
   */
  deserializeFromWASM: function(wasmMemory, ptr, count) {
    const serializer = new FloatSerializer();
    const size = count * 12; // 12 bytes per float (8 + 4 checksum)
    const buffer = wasmMemory.buffer.slice(ptr, ptr + size);
    
    return serializer.deserializeFloatArray(buffer, count);
  }
};

/**
 * Performance benchmarking utilities
 */
const FloatSerializerBenchmark = {
  /**
   * Benchmark serialization throughput
   */
  benchmarkSerialization: function(count = 1000000) {
    const serializer = new FloatSerializer();
    const testValues = this.generateTestValues(count);
    
    const startTime = performance.now();
    const buffer = serializer.serializeFloatArray(testValues);
    const endTime = performance.now();
    
    const throughput = count / ((endTime - startTime) / 1000); // values per second
    const sizeMB = buffer.byteLength / (1024 * 1024);
    
    return {
      values: count,
      timeMS: endTime - startTime,
      throughput: throughput,
      sizeMB: sizeMB,
      speedMBps: sizeMB / ((endTime - startTime) / 1000),
      valuesPerSecond: Math.round(throughput)
    };
  },

  /**
   * Benchmark deserialization throughput
   */
  benchmarkDeserialization: function(count = 1000000) {
    const serializer = new FloatSerializer();
    const testValues = this.generateTestValues(count);
    const buffer = serializer.serializeFloatArray(testValues);
    
    const startTime = performance.now();
    const result = serializer.deserializeFloatArray(buffer, count);
    const endTime = performance.now();
    
    const throughput = count / ((endTime - startTime) / 1000);
    
    return {
      values: count,
      timeMS: endTime - startTime,
      throughput: throughput,
      valuesPerSecond: Math.round(throughput)
    };
  },

  /**
   * Generate diverse test values for benchmarking
   */
  generateTestValues: function(count) {
    const values = new Array(count);
    
    for (let i = 0; i < count; i++) {
      const index = i % 10;
      switch (index) {
        case 0: // Normal values
          values[i] = Math.random() * 1000 - 500;
          break;
        case 1: // Small values
          values[i] = Math.random() * 1e-10;
          break;
        case 2: // Large values
          values[i] = Math.random() * 1e10;
          break;
        case 3: // Denormals/subnormals
          values[i] = Math.pow(2, -1074) * (i + 1);
          break;
        case 4: // Infinity
          values[i] = i % 2 === 0 ? Infinity : -Infinity;
          break;
        case 5: // NaN
          values[i] = NaN;
          break;
        case 6: // Negative values
          values[i] = -(Math.random() * 1000);
          break;
        case 7: // Very small negatives
          values[i] = -(Math.random() * 1e-10);
          break;
        case 8: // Pi and mathematical constants
          values[i] = Math.PI * (i + 1);
          break;
        case 9: // Edge case near limits
          values[i] = Number.MAX_VALUE / (i + 1);
          break;
      }
    }
    
    return values;
  }
};

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    FloatSerializer,
    WASMFloatSerializer,
    FloatSerializerBenchmark
  };
} else if (typeof window !== 'undefined') {
  window.FloatSerialization = {
    FloatSerializer,
    WASMFloatSerializer,
    FloatSerializerBenchmark
  };
}