# f64 Memory Model and Lossless Wire Contract v1.0.0

**Version**: 1.0.0  
**ABI Version**: 100  
**Precision**: IEEE 754 binary64  
**Alignment**: 16 bytes (2 x f64)  
**Date**: 2026-04-18  

---

## 1. Purpose

This document specifies the complete memory layout for all double-precision
buffers that cross the JS-to-WASM boundary, and defines the wire contract for
lossless 64-bit float serialization. It covers:

- Structure-of-arrays (SoA) arena layout for ring-buffer price/volume series
- Vector lane alignment for f64x2 SIMD operations
- Scratch buffer allocation discipline
- JSON serialization rules that preserve full 64-bit fidelity
- Mandatory binary transport zones vs. permitted JSON zones

---

## 2. Arena Memory Layout

All hot-path data resides in a single contiguous `WebAssembly.Memory` region.
The arena uses **structure-of-arrays** layout exclusively:

```
Arena (f64 slot indices):

[0..5]          CONTROL header              (6 f64 slots)
[6..6+2M)      SERIES_META                 (2 f64 slots per series, M = maxSeries)
[6+2M..6+2M+M*S)   PRICE_SLABS            (S = samplesPerSeries slots per series)
[6+2M+M*S..6+2M+2M*S) VOLUME_SLABS        (S slots per series)
[6+2M+2M*S..6+2M+2M*S+M*F) FEATURE_SLABS  (F = FEATURE_COUNT = 6 slots per series)
```

### 2.1 Control Header (slots 0-5)

| Slot | Name             | Type | Description                          |
|------|------------------|------|--------------------------------------|
| 0    | maxSeries        | f64  | Maximum number of series             |
| 1    | samplesPerSeries | f64  | Ring buffer depth per series         |
| 2    | featureCount     | f64  | Always 6                             |
| 3    | lastSeriesId     | f64  | Last series written (or -1)          |
| 4    | lastPrice        | f64  | Last price ingested                  |
| 5    | lastVolume       | f64  | Last volume ingested                 |

### 2.2 Per-Series Metadata (2 f64 slots per series)

| Offset | Name        | Type | Description                                     |
|--------|-------------|------|-------------------------------------------------|
| +0     | writeIndex  | f64  | Next write position in ring (wraps to 0 at S)   |
| +1     | sampleCount | f64  | Number of samples stored (saturates at S)       |

For series `k`, metadata starts at slot `6 + 2*k`.

### 2.3 Price Slabs

For series `k`, price data occupies `S` contiguous f64 slots starting at:
```
slot = 6 + 2*M + k*S
```

### 2.4 Volume Slabs

For series `k`, volume data occupies `S` contiguous f64 slots starting at:
```
slot = 6 + 2*M + M*S + k*S
```

### 2.5 Feature Output Slabs

For series `k`, the 6-feature output vector occupies `FEATURE_COUNT` contiguous
f64 slots starting at:
```
slot = 6 + 2*M + 2*M*S + k*FEATURE_COUNT
```

### 2.6 Total Arena Size

```
totalSlots = 6 + 2*M + M*S + M*S + M*F
           = 6 + 2*M + 2*M*S + M*F
totalBytes = totalSlots * 8
```

---

## 3. SIMD Vector Lane Alignment

WASM SIMD f64x2 lanes operate on pairs of adjacent f64 values (16 bytes).

### 3.1 Alignment Rules

- The arena base pointer is **16-byte aligned** (guaranteed by WebAssembly.Memory).
- Each series slab starts at a f64 slot index. For SIMD-safe traversal,
  the ring-buffer size `samplesPerSeries` should be **even** so that full
  f64x2 load/store pairs cover each slab without crossing slab boundaries.
- The control header (6 slots) and per-series metadata (2 slots per series)
  are both even-length, preserving alignment into the price slabs.
- The adapter rounds `samplesPerSeries` to an even number when computing
  offsets in SIMD mode.

### 3.2 Vector Lane Mapping for f64x2

For a series slab of length `S` (even), SIMD lanes map as:

```
lane pair 0:  slot[0], slot[1]
lane pair 1:  slot[2], slot[3]
...
lane pair S/2-1: slot[S-2], slot[S-1]
```

Feature vectors (6 slots) map to 3 f64x2 lane pairs:
```
pair 0: mean_price, price_variance
pair 1: vwap, latest_price
pair 2: price_momentum, mean_volume
```

---

## 4. Scratch Buffer Discipline

The scratch buffer is a contiguous region of f64-aligned memory following the
arena. It is used only for temporary staging during WASM calls.

### 4.1 Layout

```
scratchByteOffset = alignUp(arenaBytes, 16)
scratchBytes = scratchFloat64Slots * 8
```

### 4.2 Allocation Rules

- Scratch is **bump-allocated** per call and reset before each kernel invocation.
- Each allocation is 8-byte aligned (f64 alignment).
- Scratch is **never** used for persistent state. Data in scratch is valid only
  for the duration of a single kernel call.
- The adapter copies external Float64Array inputs into scratch when the input
  buffer is not backed by the same WebAssembly.Memory.

### 4.3 Capacity

Minimum scratch capacity:
```
max(4096, maxSeries * FEATURE_COUNT * 2, samplesPerSeries * 4) f64 slots
```

---

## 5. Wire Contract: Binary Transport Zones

### 5.1 Mandatory Binary Transport

Binary (Float64Array backed by WebAssembly.Memory or SharedArrayBuffer) is
**mandatory** for:

| Zone             | Direction       | Format                    | Rationale                              |
|------------------|-----------------|---------------------------|----------------------------------------|
| Arena state      | Bidirectional   | Raw f64 memory            | Hot-path data; never leaves WASM mem   |
| Price slabs      | Adapter->Kernel | Float64Array view         | Ring buffer writes                     |
| Volume slabs     | Adapter->Kernel | Float64Array view         | Ring buffer writes                     |
| Feature outputs  | Kernel->Adapter | Float64Array view         | Fixed-stride reduction writes          |
| Portfolio vecs   | Kernel->Adapter | Float64Array views        | Deviation/harvest/rebalance results    |
| Defect results   | Kernel->Adapter | Float64Array view         | 3-slot result                          |
| Regime results   | Kernel->Adapter | Float64Array view         | 4-slot result                          |
| Batch inputs     | JS->Adapter     | Float64Array / Uint32Array| Typed staging before dispatch          |
| Scratch staging  | Adapter->Kernel | Float64Array in WASM mem  | Temporary copy for non-resident inputs |

**No JSON encoding may occur in any mandatory binary zone.** JSON conversion
of any f64 value in these zones introduces potential precision loss and is
explicitly forbidden by this contract.

### 5.2 Permitted JSON Zones

JSON transport is permitted **only** for:

| Zone                     | Direction        | Encoding Rules                    |
|--------------------------|------------------|-----------------------------------|
| API request/response     | External->Bridge | Must use lossless codec (see §6)  |
| Logging / telemetry      | Bridge->External | May use lossy display format      |
| Configuration / metadata | External->Bridge | String-keyed, numeric values must use lossless codec if they contain f64 data |
| Diagnostic snapshots     | Bridge->External | Must use lossless codec for f64 values |

When JSON is used in a permitted zone, all f64 values **must** be encoded using
the lossless wire codec defined in §6 below. The codec is the **only** approved
serialization for IEEE 754 binary64 values that need to survive a JSON round
trip.

### 5.3 Forbidden Precision Downgrade Points

The following are **explicitly forbidden**:

1. `JSON.stringify(value)` for any f64 that must survive roundtrip — JS
   `JSON.stringify` uses a "shortest representation" algorithm that may not
   preserve the exact bit pattern for values near the precision boundary.

2. `Number(string)` for any f64 read from JSON — may produce a different
   bit pattern than the original value.

3. Passing f64 values through `postMessage` structured clone without
   verifying bit preservation (structured clone preserves f64 exactly, but
   the consumer must not re-serialize via `JSON.stringify`).

4. Storing f64 values as `number` in plain JS objects that will later be
   serialized to JSON without the lossless codec.

---

## 6. Lossless Wire Codec

### 6.1 Canonical Encoding

Every f64 value is encoded as one of:

| Category         | JSON representation                     | Example                           |
|------------------|-----------------------------------------|-----------------------------------|
| Finite normal    | `{"_f64": "<hex16>"}`                   | `{"_f64": "400921fb54442d18"}`    |
| Finite subnormal | `{"_f64": "<hex16>"}`                   | `{"_f64": "0000000000000001"}`    |
| +Infinity        | `{"_f64": "7ff0000000000000"}`          |                                   |
| -Infinity        | `{"_f64": "fff0000000000000"}`          |                                   |
| +0               | `{"_f64": "0000000000000000"}`          |                                   |
| -0               | `{"_f64": "8000000000000000"}`          |                                   |
| Canonical NaN    | `{"_f64": "7ff8000000000000"}`          |                                   |
| Quiet NaN (any)  | `{"_f64": "<hex16>"}` with NaN pattern  |                                   |
| Signaling NaN    | `{"_f64": "<hex16>"}`                   |                                   |

The hex16 string is the **big-endian** hexadecimal representation of the 64-bit
IEEE 754 bit pattern stored as a `BigUint64` in the platform's native byte order.

### 6.2 Array Encoding

A `Float64Array` is encoded as:

```json
{
  "_f64vec": ["<hex16>", "<hex16>", ...],
  "_length": <number of elements>
}
```

This avoids per-element object overhead while preserving each f64 bit pattern.

### 6.3 NaN Payload Policy

- The codec preserves the **full 64-bit NaN bit pattern** including the sign
  bit, quiet/signaling bit, and payload bits.
- When constructing NaN values in JS, `Number.NaN` produces the canonical
  quiet NaN (`0x7ff8000000000000`). The codec can represent any NaN pattern.
- On decode, the codec uses `DataView.getFloat64()` to reconstruct the exact
  bit pattern, avoiding JS `Number()` constructor precision loss.

### 6.4 Endianness

- The hex16 string represents **big-endian** byte order (network byte order).
- On encode: `DataView.getBigUint64(0)` reads the f64 bytes in the platform's
  native order. The hex string is always the big-endian interpretation
  (MSB first), which `getBigUint64` already provides when reading in
  big-endian mode (the default).
- On decode: `DataView.setBigUint64(0, BigInt(hexString))` writes in
  big-endian order, then `DataView.getFloat64(0)` reads back the f64.
- This is correct on all platforms because `DataView` uses big-endian by
  default, and WebAssembly linear memory is little-endian, but the DataView
  I/O handles the conversion correctly.

### 6.5 Roundtrip Guarantee

For any f64 value `v`:
```
decode(encode(v)) === v  (bit-identical, including NaN and signed zero)
```

This is verified by the test suite for:
- All special values: +0, -0, +Infinity, -Infinity, canonical NaN
- Subnormal numbers (smallest positive, largest subnormal)
- Extremal normal values (MIN_NORMAL, MAX_FINITE)
- Representative normal values (0.5, 1.0, pi, e, etc.)
- All-zeros and all-ones bit patterns
- Signaling NaN variants

---

## 7. Checksum and Integrity

For integrity verification across the wire:

### 7.1 Buffer Checksum

Before serialization, a Fletcher-16 checksum is computed over the raw bytes of
each Float64Array. The checksum is included in the wire format:

```json
{
  "_f64vec": ["<hex16>", ...],
  "_length": <n>,
  "_checksum": "<4-hex-digit Fletcher-16>"
}
```

### 7.2 Verification

On deserialization, the checksum is recomputed and compared. A mismatch
indicates data corruption in transit.

---

## 8. Transport Rules Summary

```
╔══════════════════════════════════════════════════════════════════╗
║                    TRANSPORT RULES                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  JS Orchestration ──Float64Array──> WASM Memory                  ║
║       (typed staging)     |          (arena + scratch)           ║
║                           v                                     ║
║                    Numeric Core (f64 math)                       ║
║                           |                                     ║
║                           v                                     ║
║  JS Orchestration <──Float64Array──  WASM Memory                 ║
║       (result read)       |          (result slabs)              ║
║                           v                                     ║
║  External API ────lossless JSON────>  JS Bridge                  ║
║       (f64_wire_codec)    |          (handoff bridge)            ║
║                           v                                     ║
║  External API <───lossless JSON────  JS Bridge                   ║
║                                                                  ║
║  FORBIDDEN: JSON.stringify on any f64 in a binary zone           ║
║  FORBIDDEN: Number() on hex-encoded f64 strings                  ║
║  FORBIDDEN: Any precision-downgrade re-encoding                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 9. Reference: Constant Values

| Constant                   | Value                                          |
|----------------------------|-------------------------------------------------|
| ABI_VERSION                | 100                                             |
| ALIGNMENT_BYTES            | 16                                              |
| FLOAT64_BYTES              | 8                                               |
| CONTROL_SLOTS              | 6                                               |
| META_SLOTS_PER_SERIES      | 2                                               |
| FEATURE_COUNT              | 6                                               |
| FEATURE_INDICES            | {MEAN_PRICE:0, PRICE_VARIANCE:1, VWAP:2,        |
|                            |  LATEST_PRICE:3, PRICE_MOMENTUM:4, MEAN_VOLUME:5}|
| RESULT_PORTFOLIO_SLOTS     | 5                                               |
| RESULT_DEFECT_SLOTS        | 3                                               |
| RESULT_REGIME_SLOTS        | 4                                               |
| NaN_CANONICAL_HEX          | "7ff8000000000000"                               |
| +Infinity_HEX              | "7ff0000000000000"                               |
| -Infinity_HEX              | "fff0000000000000"                               |
| +0_HEX                     | "0000000000000000"                               |
| -0_HEX                     | "8000000000000000"                               |
