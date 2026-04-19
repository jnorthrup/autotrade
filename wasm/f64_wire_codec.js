'use strict';

/**
 * Lossless f64 Wire Codec for Dreamer WASM Kernel
 *
 * Provides serialization/deserialization that preserves exact IEEE 754 binary64
 * bit patterns across JSON boundaries. Every f64 value is encoded as a 16-digit
 * hex string representing the 64-bit pattern in big-endian (network) byte order.
 *
 * Edge cases handled:
 * - NaN with full payload preservation (quiet, signaling, sign bit, mantissa)
 * - +Infinity, -Infinity
 * - +0, -0 (signed zero distinction preserved)
 * - Subnormal numbers (denormals)
 * - Extremal values (MAX_FINITE, MIN_POSITIVE_NORMAL, MIN_POSITIVE_SUBNORMAL)
 */

// ============================================================================
// Constants
// ============================================================================

const FLOAT64_BYTES = 8;
const HEX_WIDTH = 16;

// Pre-allocated DataView for single-value encode/decode (not shared across calls)
const _buf = new ArrayBuffer(FLOAT64_BYTES);
const _dv = new DataView(_buf);
const _u8 = new Uint8Array(_buf);

// ============================================================================
// Single Value Encode / Decode
// ============================================================================

/**
 * Encode a single f64 value into its 16-digit hex string representation.
 *
 * The hex string is the big-endian representation of the IEEE 754 binary64
 * bit pattern. This preserves all edge cases: NaN payloads, signed zero,
 * infinities, subnormals.
 *
 * @param {number} value - Any JavaScript number (including NaN, Infinity, -0)
 * @returns {string} 16-character lowercase hex string
 */
function encodeF64(value) {
  _dv.setFloat64(0, value, false); // false = big-endian write
  const high = _dv.getUint32(0, false);
  const low = _dv.getUint32(4, false);
  return high.toString(HEX_WIDTH).padStart(8, '0') + low.toString(HEX_WIDTH).padStart(8, '0');
}

/**
 * Decode a 16-digit hex string back into an f64 value.
 *
 * This reconstructs the exact IEEE 754 bit pattern, including NaN payloads
 * and signed zero distinction.
 *
 * @param {string} hex - 16-character hex string (case-insensitive)
 * @returns {number} The reconstructed JavaScript number
 * @throws {TypeError} If hex is not a valid 16-char hex string
 */
function decodeF64(hex) {
  if (typeof hex !== 'string' || hex.length !== HEX_WIDTH || !/^[0-9a-fA-F]{16}$/.test(hex)) {
    throw new TypeError(`f64 wire codec: expected 16-char hex string, got "${hex}"`);
  }
  const high = parseInt(hex.substring(0, 8), 16) >>> 0;
  const low = parseInt(hex.substring(8, 16), 16) >>> 0;
  _dv.setUint32(0, high, false); // big-endian
  _dv.setUint32(4, low, false);
  return _dv.getFloat64(0, false); // big-endian read
}

/**
 * Check if a value is a lossless-encoded f64 object.
 *
 * @param {*} value - Any JSON value
 * @returns {boolean}
 */
function isEncodedF64(value) {
  return value !== null && typeof value === 'object' && typeof value._f64 === 'string';
}

// ============================================================================
// Array Encode / Decode
// ============================================================================

/**
 * Encode a Float64Array into a lossless JSON-compatible object.
 *
 * The output format is:
 * ```json
 * { "_f64vec": ["<hex16>", ...], "_length": <n>, "_checksum": "<4-hex>" }
 * ```
 *
 * @param {Float64Array} arr - Source array
 * @param {number} [offset=0] - Start offset
 * @param {number} [count] - Number of elements to encode (default: arr.length - offset)
 * @returns {object} JSON-serializable object with lossless f64 encoding
 */
function encodeF64Array(arr, offset = 0, count) {
  if (!(arr instanceof Float64Array)) {
    throw new TypeError('f64 wire codec: encodeF64Array requires a Float64Array');
  }
  const start = offset | 0;
  const len = (count !== undefined ? count : arr.length - start) | 0;

  if (start < 0 || len < 0 || start + len > arr.length) {
    throw new RangeError(`f64 wire codec: invalid offset/count for array of length ${arr.length}`);
  }

  const vec = new Array(len);
  // Use a dedicated buffer for bulk encode to avoid shared _buf clobbering
  const chunkBuf = new ArrayBuffer(FLOAT64_BYTES);
  const chunkDv = new DataView(chunkBuf);

  let checksumA = 0;
  let checksumB = 0;

  for (let i = 0; i < len; i++) {
    const value = arr[start + i];
    chunkDv.setFloat64(0, value, false);
    const high = chunkDv.getUint32(0, false);
    const low = chunkDv.getUint32(4, false);
    const hex = high.toString(HEX_WIDTH).padStart(8, '0') + low.toString(HEX_WIDTH).padStart(8, '0');
    vec[i] = hex;

    // Fletcher-16 over the raw 8 bytes
    const raw = new Uint8Array(chunkBuf);
    for (let b = 0; b < FLOAT64_BYTES; b++) {
      checksumA = (checksumA + raw[b]) % 255;
      checksumB = (checksumB + checksumA) % 255;
    }
  }

  const checksum = ((checksumA << 8) | checksumB).toString(16).padStart(4, '0');

  return Object.freeze({
    _f64vec: Object.freeze(vec),
    _length: len,
    _checksum: checksum,
  });
}

/**
 * Decode a lossless-encoded JSON object back into a Float64Array.
 *
 * @param {object} encoded - Object produced by encodeF64Array
 * @param {Float64Array} [out] - Optional pre-allocated output array
 * @returns {Float64Array} Decoded array
 * @throws {TypeError} If encoded format is invalid
 * @throws {Error} If checksum verification fails
 */
function decodeF64Array(encoded, out) {
  if (!encoded || !Array.isArray(encoded._f64vec)) {
    throw new TypeError('f64 wire codec: decodeF64Array requires an object with _f64vec array');
  }

  const vec = encoded._f64vec;
  const len = typeof encoded._length === 'number' ? encoded._length : vec.length;
  const expectedChecksum = encoded._checksum;

  if (vec.length !== len) {
    throw new TypeError(`f64 wire codec: _f64vec length ${vec.length} does not match _length ${len}`);
  }

  const result = out instanceof Float64Array && out.length >= len
    ? out.subarray(0, len)
    : new Float64Array(len);

  const chunkBuf = new ArrayBuffer(FLOAT64_BYTES);
  const chunkDv = new DataView(chunkBuf);

  let checksumA = 0;
  let checksumB = 0;

  for (let i = 0; i < len; i++) {
    const hex = vec[i];
    if (typeof hex !== 'string' || hex.length !== HEX_WIDTH || !/^[0-9a-fA-F]{16}$/.test(hex)) {
      throw new TypeError(`f64 wire codec: invalid hex at index ${i}: "${hex}"`);
    }
    const high = parseInt(hex.substring(0, 8), 16) >>> 0;
    const low = parseInt(hex.substring(8, 16), 16) >>> 0;
    chunkDv.setUint32(0, high, false);
    chunkDv.setUint32(4, low, false);
    result[i] = chunkDv.getFloat64(0, false);

    // Verify checksum during decode
    const raw = new Uint8Array(chunkBuf);
    for (let b = 0; b < FLOAT64_BYTES; b++) {
      checksumA = (checksumA + raw[b]) % 255;
      checksumB = (checksumB + checksumA) % 255;
    }
  }

  if (typeof expectedChecksum === 'string') {
    const actualChecksum = ((checksumA << 8) | checksumB).toString(16).padStart(4, '0');
    if (actualChecksum !== expectedChecksum) {
      throw new Error(
        `f64 wire codec: checksum mismatch — expected ${expectedChecksum}, got ${actualChecksum}`
      );
    }
  }

  return result;
}

// ============================================================================
// Bit-Level Comparison
// ============================================================================

/**
 * Compare two f64 values for bit-identical equality.
 *
 * Unlike `===`, this distinguishes:
 * - +0 from -0
 * - Different NaN bit patterns (as far as JS can observe them)
 * - NaN from NaN (returns true for same-bits NaN)
 *
 * @param {number} a
 * @param {number} b
 * @returns {boolean} True if bit-identical
 */
function bitIdentical(a, b) {
  const dv = new DataView(new ArrayBuffer(16));
  dv.setFloat64(0, a, false);
  dv.setFloat64(8, b, false);
  return dv.getUint32(0, false) === dv.getUint32(8, false)
      && dv.getUint32(4, false) === dv.getUint32(12, false);
}

/**
 * Compare two Float64Arrays for bit-identical element-wise equality.
 *
 * @param {Float64Array} a
 * @param {Float64Array} b
 * @returns {{equal: boolean, firstDiffIndex: number|null}}
 */
function arrayBitIdentical(a, b) {
  if (a.length !== b.length) {
    return { equal: false, firstDiffIndex: 0 };
  }
  for (let i = 0; i < a.length; i++) {
    if (!bitIdentical(a[i], b[i])) {
      return { equal: false, firstDiffIndex: i };
    }
  }
  return { equal: true, firstDiffIndex: null };
}

// ============================================================================
// Roundtrip Verification
// ============================================================================

/**
 * Verify that a single value survives a complete encode-decode roundtrip
 * with bit-identical fidelity.
 *
 * @param {number} value
 * @returns {{pass: boolean, original: string, recovered: string}}
 */
function verifyRoundtrip(value) {
  const hex = encodeF64(value);
  const recovered = decodeF64(hex);
  const pass = bitIdentical(value, recovered);
  return {
    pass,
    original: hex,
    recovered: encodeF64(recovered),
  };
}

/**
 * Verify that a Float64Array survives a complete encode-decode roundtrip.
 *
 * @param {Float64Array} arr
 * @returns {{pass: boolean, firstDiffIndex: number|null, checksumValid: boolean}}
 */
function verifyArrayRoundtrip(arr) {
  const encoded = encodeF64Array(arr);
  const decoded = decodeF64Array(encoded);
  const cmp = arrayBitIdentical(arr, decoded);
  return {
    pass: cmp.equal,
    firstDiffIndex: cmp.firstDiffIndex,
    checksumValid: true, // decodeF64Array throws on checksum mismatch
  };
}

// ============================================================================
// JSON Serialization Helpers
// ============================================================================

/**
 * JSON.stringify replacer function for lossless f64 encoding.
 *
 * Usage:
 *   JSON.stringify({price: 1.1, vol: 2.2}, f64Replacer)
 *
 * All number values are encoded as {"_f64": "<hex>"}.
 */
function f64Replacer(key, value) {
  if (typeof value === 'number') {
    return { _f64: encodeF64(value) };
  }
  if (value instanceof Float64Array) {
    return encodeF64Array(value);
  }
  return value;
}

/**
 * JSON.parse reviver function for lossless f64 decoding.
 *
 * Usage:
 *   JSON.parse(json, f64Reviver)
 *
 * Objects with _f64 are decoded back to exact numbers.
 * Objects with _f64vec are decoded back to Float64Array (converted to Array
 * for JSON compatibility).
 */
function f64Reviver(key, value) {
  if (value !== null && typeof value === 'object') {
    if (typeof value._f64 === 'string') {
      return decodeF64(value._f64);
    }
    if (Array.isArray(value._f64vec)) {
      return decodeF64Array(value);
    }
  }
  return value;
}

/**
 * Stringify an object with lossless f64 encoding.
 */
function losslessStringify(obj, space) {
  return JSON.stringify(obj, f64Replacer, space);
}

/**
 * Parse a lossless-encoded JSON string.
 */
function losslessParse(json) {
  return JSON.parse(json, f64Reviver);
}

// ============================================================================
// Known Edge Case Values
// ============================================================================

const EDGE_CASES = Object.freeze({
  POSITIVE_ZERO: 0.0,
  NEGATIVE_ZERO: -0.0,
  POSITIVE_INFINITY: Infinity,
  NEGATIVE_INFINITY: -Infinity,
  CANONICAL_NAN: NaN,
  MAX_FINITE: (2.0 - Math.pow(2, -52)) * Math.pow(2, 1023),
  MIN_POSITIVE_NORMAL: Math.pow(2, -1022),
  MIN_POSITIVE_SUBNORMAL: Math.pow(2, -1074),
  LARGEST_SUBNORMAL: (1.0 - Math.pow(2, -52)) * Math.pow(2, -1022),
  NEGATIVE_MAX_FINITE: -(2.0 - Math.pow(2, -52)) * Math.pow(2, 1023),
  NEGATIVE_MIN_NORMAL: -Math.pow(2, -1022),
  ONE: 1.0,
  NEGATIVE_ONE: -1.0,
  HALF: 0.5,
  PI: Math.PI,
  E: Math.E,
  SQRT2: Math.SQRT2,
  LN2: Math.LN2,
  LN10: Math.LN10,
  EPSILON: Math.pow(2, -52),
  // Near-boundary values that expose JSON.stringify precision issues
  NEAR_PRECISION_LIMIT: 1.0000000000000002,
  HIGH_PRECISION_PRICE: 123456.78901234567,
  VERY_SMALL_POSITIVE: 1e-308,
  VERY_LARGE_POSITIVE: 1e308,
});

// Pre-computed hex encodings for known special values
const KNOWN_HEX = Object.freeze({
  POSITIVE_ZERO: '0000000000000000',
  NEGATIVE_ZERO: '8000000000000000',
  POSITIVE_INFINITY: '7ff0000000000000',
  NEGATIVE_INFINITY: 'fff0000000000000',
  CANONICAL_NAN: '7ff8000000000000',
  ONE: '3ff0000000000000',
  NEGATIVE_ONE: 'bff0000000000000',
  HALF: '3fe0000000000000',
});

// ============================================================================
// Construct NaN values with specific bit patterns
// ============================================================================

/**
 * Construct an f64 value with a specific 64-bit pattern given as a hex string.
 */
function fromHex(hex) {
  return decodeF64(hex);
}

/**
 * Get the hex representation of an f64 value.
 */
function toHex(value) {
  return encodeF64(value);
}

/**
 * Construct a quiet NaN with a specific payload.
 * The quiet NaN bit pattern is: sign(1) | exponent(11 x 1) | quiet_bit(1) | payload(51 bits)
 *
 * @param {number} signBit - 0 or 1
 * @param {number} payload - 51-bit integer payload
 * @returns {number} The NaN value
 */
function makeQuietNaN(signBit, payload) {
  const sign = signBit & 1;
  const payloadBits = BigInt(payload) & 0x7FFFFFFFFFFFFn;
  const bits = (BigInt(sign) << 63n) | 0x7FF8000000000000n | payloadBits;
  const buf = new ArrayBuffer(FLOAT64_BYTES);
  const dv = new DataView(buf);
  dv.setBigUint64(0, bits, false);
  return dv.getFloat64(0, false);
}

/**
 * Construct a signaling NaN with a specific payload.
 *
 * @param {number} signBit - 0 or 1
 * @param {number} payload - 51-bit integer payload (must have bit 51 = 0)
 * @returns {number} The NaN value
 */
function makeSignalingNaN(signBit, payload) {
  const sign = signBit & 1;
  const payloadBits = (BigInt(payload) & 0x7FFFFFFFFFFFFn);
  // Signaling NaN: exponent all 1s, quiet bit (bit 51) = 0, payload nonzero
  const bits = (BigInt(sign) << 63n) | 0x7FF0000000000000n | payloadBits;
  if (payloadBits === 0n) {
    throw new RangeError('signaling NaN requires a nonzero payload');
  }
  const buf = new ArrayBuffer(FLOAT64_BYTES);
  const dv = new DataView(buf);
  dv.setBigUint64(0, bits, false);
  return dv.getFloat64(0, false);
}

// ============================================================================
// Fletcher-16 Checksum (standalone, for external use)
// ============================================================================

/**
 * Compute a Fletcher-16 checksum over a byte array.
 *
 * @param {Uint8Array} bytes
 * @returns {string} 4-digit hex checksum
 */
function fletcher16(bytes) {
  let a = 0;
  let b = 0;
  for (let i = 0; i < bytes.length; i++) {
    a = (a + bytes[i]) % 255;
    b = (b + a) % 255;
  }
  return ((a << 8) | b).toString(16).padStart(4, '0');
}

// ============================================================================
// Exports
// ============================================================================

module.exports = Object.freeze({
  // Single value codec
  encodeF64,
  decodeF64,
  isEncodedF64,

  // Array codec
  encodeF64Array,
  decodeF64Array,

  // Bit comparison
  bitIdentical,
  arrayBitIdentical,

  // Roundtrip verification
  verifyRoundtrip,
  verifyArrayRoundtrip,

  // JSON integration
  f64Replacer,
  f64Reviver,
  losslessStringify,
  losslessParse,

  // Hex utilities
  fromHex,
  toHex,

  // NaN construction
  makeQuietNaN,
  makeSignalingNaN,

  // Checksum
  fletcher16,

  // Known values
  EDGE_CASES,
  KNOWN_HEX,
});
