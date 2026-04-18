'use strict';

const { VALIDATION_PLAN } = require('./validation_plan');

const FLOAT64_BYTES = Float64Array.BYTES_PER_ELEMENT;
const CONTROL_SLOTS = 6;
const META_SLOTS_PER_SERIES = 2;
const FEATURE_COUNT = 6;
const FEATURE_NAMES = Object.freeze([
  'mean_price',
  'price_variance',
  'vwap',
  'latest_price',
  'price_momentum',
  'mean_volume',
]);

function asPositiveInteger(value, label) {
  if (!Number.isFinite(value) || Math.trunc(value) !== value) {
    throw new TypeError(`${label} must be a finite integer`);
  }
  if (value < 1) {
    throw new RangeError(`${label} must be >= 1`);
  }
  return value;
}

function asNonNegativeInteger(value, label) {
  if (!Number.isFinite(value) || Math.trunc(value) !== value) {
    throw new TypeError(`${label} must be a finite integer`);
  }
  if (value < 0) {
    throw new RangeError(`${label} must be >= 0`);
  }
  return value;
}

function asSeriesId(value, maxSeries) {
  if (!Number.isFinite(value) || Math.trunc(value) !== value) {
    throw new TypeError('seriesId must be a finite integer');
  }
  if (value < 0 || value >= maxSeries) {
    throw new RangeError(`seriesId must be in [0, ${maxSeries - 1}]`);
  }
  return value;
}

function asFiniteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number`);
  }
  return value;
}

function asFloat64Array(value, label) {
  if (!(value instanceof Float64Array)) {
    throw new TypeError(`${label} must be a Float64Array`);
  }
  return value;
}

function computeLayout(maxSeries, samplesPerSeries) {
  const seriesMetaStart = CONTROL_SLOTS;
  const seriesMetaEnd = seriesMetaStart + maxSeries * META_SLOTS_PER_SERIES;
  const pricesStart = seriesMetaEnd;
  const pricesEnd = pricesStart + maxSeries * samplesPerSeries;
  const volumesStart = pricesEnd;
  const volumesEnd = volumesStart + maxSeries * samplesPerSeries;
  const featureStart = volumesEnd;
  const featureEnd = featureStart + maxSeries * FEATURE_COUNT;

  return Object.freeze({
    controlStart: 0,
    controlEnd: CONTROL_SLOTS,
    seriesMetaStart,
    seriesMetaEnd,
    pricesStart,
    pricesEnd,
    volumesStart,
    volumesEnd,
    featureStart,
    featureEnd,
    totalSlots: featureEnd,
  });
}

function createArena(totalSlots) {
  const slots = asPositiveInteger(totalSlots, 'totalSlots');
  const buffer = new ArrayBuffer(slots * FLOAT64_BYTES);
  const f64 = new Float64Array(buffer);
  return Object.freeze({ buffer, f64, totalSlots: slots });
}

class DoubleVectorKernel {
  constructor(maxSeries, samplesPerSeries, arena) {
    this.maxSeries = asPositiveInteger(maxSeries, 'maxSeries');
    this.samplesPerSeries = asPositiveInteger(samplesPerSeries, 'samplesPerSeries');
    this.layout = computeLayout(this.maxSeries, this.samplesPerSeries);
    this.arena = arena || createArena(this.layout.totalSlots);

    if (!(this.arena && this.arena.f64 instanceof Float64Array)) {
      throw new TypeError('arena must expose a Float64Array as arena.f64');
    }
    if (this.arena.f64.length < this.layout.totalSlots) {
      throw new RangeError(
        `arena is too small: need ${this.layout.totalSlots} Float64 slots, got ${this.arena.f64.length}`,
      );
    }

    const f64 = this.arena.f64;
    this.control = f64.subarray(this.layout.controlStart, this.layout.controlEnd);
    this.seriesMeta = f64.subarray(this.layout.seriesMetaStart, this.layout.seriesMetaEnd);
    this.prices = f64.subarray(this.layout.pricesStart, this.layout.pricesEnd);
    this.volumes = f64.subarray(this.layout.volumesStart, this.layout.volumesEnd);
    this.featureSlab = f64.subarray(this.layout.featureStart, this.layout.featureEnd);
    this.priceViews = Array.from({ length: this.maxSeries }, (_, seriesId) => (
      f64.subarray(
        this.layout.pricesStart + seriesId * this.samplesPerSeries,
        this.layout.pricesStart + (seriesId + 1) * this.samplesPerSeries,
      )
    ));
    this.volumeViews = Array.from({ length: this.maxSeries }, (_, seriesId) => (
      f64.subarray(
        this.layout.volumesStart + seriesId * this.samplesPerSeries,
        this.layout.volumesStart + (seriesId + 1) * this.samplesPerSeries,
      )
    ));
    this.outputViews = Array.from({ length: this.maxSeries }, (_, seriesId) => (
      f64.subarray(
        this.layout.featureStart + seriesId * FEATURE_COUNT,
        this.layout.featureStart + (seriesId + 1) * FEATURE_COUNT,
      )
    ));

    this.control[0] = this.maxSeries;
    this.control[1] = this.samplesPerSeries;
    this.control[2] = FEATURE_COUNT;
    this.control[3] = 0;
    this.control[4] = Number.NaN;
    this.control[5] = Number.NaN;
  }

  _seriesMetaIndex(seriesId) {
    return seriesId * META_SLOTS_PER_SERIES;
  }

  _seriesBaseOffset(seriesId) {
    return seriesId * this.samplesPerSeries;
  }

  _requireSeriesId(seriesId) {
    return asSeriesId(seriesId, this.maxSeries);
  }

  ingestTick(seriesId, price, volume) {
    const sid = this._requireSeriesId(seriesId);
    const p = asFiniteNumber(price, 'price');
    const v = asFiniteNumber(volume, 'volume');

    const metaIndex = this._seriesMetaIndex(sid);
    const writeIndex = Math.trunc(this.seriesMeta[metaIndex]);
    const sampleCount = Math.trunc(this.seriesMeta[metaIndex + 1]);
    const baseOffset = this._seriesBaseOffset(sid);
    const slot = baseOffset + writeIndex;

    this.prices[slot] = p;
    this.volumes[slot] = v;

    const nextWriteIndex = writeIndex + 1;
    this.seriesMeta[metaIndex] = nextWriteIndex === this.samplesPerSeries ? 0 : nextWriteIndex;
    this.seriesMeta[metaIndex + 1] = sampleCount < this.samplesPerSeries
      ? sampleCount + 1
      : this.samplesPerSeries;
    this.control[3] = sid;
    this.control[4] = p;
    this.control[5] = v;

    return this.seriesMeta[metaIndex + 1];
  }

  ingestBatch(seriesId, prices, volumes, priceOffset = 0, volumeOffset = 0, count = prices.length - priceOffset) {
    const sid = this._requireSeriesId(seriesId);
    const priceSeries = asFloat64Array(prices, 'prices');
    const volumeSeries = asFloat64Array(volumes, 'volumes');
    const sourcePriceOffset = asNonNegativeInteger(priceOffset, 'priceOffset');
    const sourceVolumeOffset = asNonNegativeInteger(volumeOffset, 'volumeOffset');
    const batchCount = asNonNegativeInteger(count, 'count');

    if (sourcePriceOffset + batchCount > priceSeries.length) {
      throw new RangeError('prices does not contain enough samples for the requested batch');
    }
    if (sourceVolumeOffset + batchCount > volumeSeries.length) {
      throw new RangeError('volumes does not contain enough samples for the requested batch');
    }
    if (batchCount === 0) {
      return this.seriesMeta[this._seriesMetaIndex(sid) + 1];
    }
    if (batchCount > this.samplesPerSeries) {
      throw new RangeError(
        `count must be <= samplesPerSeries (${this.samplesPerSeries}) so the ring buffer can stay contiguous in at most two spans`,
      );
    }

    const metaIndex = this._seriesMetaIndex(sid);
    let writeIndex = Math.trunc(this.seriesMeta[metaIndex]);
    let sampleCount = Math.trunc(this.seriesMeta[metaIndex + 1]);
    const baseOffset = this._seriesBaseOffset(sid);
    const firstSpan = Math.min(batchCount, this.samplesPerSeries - writeIndex);
    let dest = baseOffset + writeIndex;
    let sourceIndex = 0;

    for (let i = 0; i < firstSpan; i += 1) {
      this.prices[dest + i] = priceSeries[sourcePriceOffset + sourceIndex + i];
      this.volumes[dest + i] = volumeSeries[sourceVolumeOffset + sourceIndex + i];
    }

    sourceIndex += firstSpan;
    const secondSpan = batchCount - firstSpan;
    if (secondSpan > 0) {
      dest = baseOffset;
      for (let i = 0; i < secondSpan; i += 1) {
        this.prices[dest + i] = priceSeries[sourcePriceOffset + sourceIndex + i];
        this.volumes[dest + i] = volumeSeries[sourceVolumeOffset + sourceIndex + i];
      }
    }

    writeIndex += batchCount;
    if (writeIndex >= this.samplesPerSeries) {
      writeIndex -= this.samplesPerSeries;
    }
    sampleCount += batchCount;
    if (sampleCount > this.samplesPerSeries) {
      sampleCount = this.samplesPerSeries;
    }

    this.seriesMeta[metaIndex] = writeIndex;
    this.seriesMeta[metaIndex + 1] = sampleCount;
    this.control[3] = sid;
    this.control[4] = priceSeries[sourcePriceOffset + batchCount - 1];
    this.control[5] = volumeSeries[sourceVolumeOffset + batchCount - 1];

    return sampleCount;
  }

  writeWindowFeatures(seriesId, windowSize, out, outOffset = 0) {
    const sid = this._requireSeriesId(seriesId);
    const requestedWindow = asPositiveInteger(windowSize, 'windowSize');

    if (!(out instanceof Float64Array)) {
      throw new TypeError('out must be a Float64Array');
    }
    if (!Number.isFinite(outOffset) || Math.trunc(outOffset) !== outOffset || outOffset < 0) {
      throw new TypeError('outOffset must be a non-negative integer');
    }
    if (out.length - outOffset < FEATURE_COUNT) {
      throw new RangeError(`out needs at least ${FEATURE_COUNT} Float64 slots starting at outOffset`);
    }

    const metaIndex = this._seriesMetaIndex(sid);
    const writeIndex = Math.trunc(this.seriesMeta[metaIndex]);
    const sampleCount = Math.trunc(this.seriesMeta[metaIndex + 1]);
    const usableWindow = Math.min(sampleCount, requestedWindow);

    if (usableWindow === 0) {
      for (let i = 0; i < FEATURE_COUNT; i += 1) {
        out[outOffset + i] = Number.NaN;
      }
      return out;
    }

    const baseOffset = this._seriesBaseOffset(sid);
    const windowStart = writeIndex >= usableWindow
      ? writeIndex - usableWindow
      : this.samplesPerSeries + writeIndex - usableWindow;
    const firstSpan = Math.min(usableWindow, this.samplesPerSeries - windowStart);

    let sumPrice = 0;
    let sumPriceSq = 0;
    let sumVolume = 0;
    let notional = 0;
    let oldestPrice = 0;
    let latestPrice = 0;

    let index = baseOffset + windowStart;
    let price = this.prices[index];
    let volume = this.volumes[index];
    oldestPrice = price;
    latestPrice = price;
    sumPrice = price;
    sumPriceSq = price * price;
    sumVolume = volume;
    notional = price * volume;

    for (let i = 1; i < firstSpan; i += 1) {
      price = this.prices[index + i];
      volume = this.volumes[index + i];
      latestPrice = price;
      sumPrice += price;
      sumPriceSq += price * price;
      sumVolume += volume;
      notional += price * volume;
    }

    const secondSpan = usableWindow - firstSpan;
    if (secondSpan > 0) {
      index = baseOffset;
      price = this.prices[index];
      volume = this.volumes[index];
      latestPrice = price;
      sumPrice += price;
      sumPriceSq += price * price;
      sumVolume += volume;
      notional += price * volume;

      for (let i = 1; i < secondSpan; i += 1) {
        price = this.prices[index + i];
        volume = this.volumes[index + i];
        latestPrice = price;
        sumPrice += price;
        sumPriceSq += price * price;
        sumVolume += volume;
        notional += price * volume;
      }
    }

    const meanPrice = sumPrice / usableWindow;
    let variance = sumPriceSq / usableWindow - (meanPrice * meanPrice);
    if (variance < 0 && variance > -1e-12) {
      variance = 0;
    }
    const vwap = sumVolume === 0 ? Number.NaN : notional / sumVolume;
    const priceMomentum = latestPrice - oldestPrice;
    const meanVolume = sumVolume / usableWindow;

    out[outOffset + 0] = meanPrice;
    out[outOffset + 1] = variance;
    out[outOffset + 2] = vwap;
    out[outOffset + 3] = latestPrice;
    out[outOffset + 4] = priceMomentum;
    out[outOffset + 5] = meanVolume;
    return out;
  }

  evaluateBatch(seriesIds, windowSizes, out, outOffset = 0, count = seriesIds.length) {
    const sidSeries = asFloat64Array(seriesIds, 'seriesIds');
    const windowSeries = asFloat64Array(windowSizes, 'windowSizes');
    const batchCount = asNonNegativeInteger(count, 'count');

    if (sidSeries.length < batchCount) {
      throw new RangeError('seriesIds does not contain enough entries for the requested batch');
    }
    if (windowSeries.length < batchCount) {
      throw new RangeError('windowSizes does not contain enough entries for the requested batch');
    }
    if (!(out instanceof Float64Array)) {
      throw new TypeError('out must be a Float64Array');
    }
    if (!Number.isFinite(outOffset) || Math.trunc(outOffset) !== outOffset || outOffset < 0) {
      throw new TypeError('outOffset must be a non-negative integer');
    }
    if (out.length - outOffset < batchCount * FEATURE_COUNT) {
      throw new RangeError(`out needs at least ${batchCount * FEATURE_COUNT} Float64 slots starting at outOffset`);
    }

    for (let i = 0; i < batchCount; i += 1) {
      this.writeWindowFeatures(
        sidSeries[i],
        windowSeries[i],
        out,
        outOffset + (i * FEATURE_COUNT),
      );
    }

    return out;
  }

  readPriceView(seriesId) {
    const sid = this._requireSeriesId(seriesId);
    return this.priceViews[sid];
  }

  readVolumeView(seriesId) {
    const sid = this._requireSeriesId(seriesId);
    return this.volumeViews[sid];
  }

  getOutputView(seriesId) {
    const sid = this._requireSeriesId(seriesId);
    return this.outputViews[sid];
  }
}

const KERNEL_CONTRACTS = Object.freeze({
  ingestTick: Object.freeze({
    name: 'ingestTick',
    shape: 'single-tick write into contiguous SoA ring buffers',
    layout: 'prices[series][slot], volumes[series][slot], metadata[series][writeIndex/sampleCount]',
    ringBufferRule: 'each series owns a fixed-size circular buffer; writes advance a monotonically wrapped index',
    hotPathRequirements: Object.freeze([
      'numeric scalars only',
      'no object lookups',
      'no per-tick allocation',
      'no mixed numeric types',
    ]),
  }),
  ingestBatch: Object.freeze({
    name: 'ingestBatch',
    shape: 'batched contiguous writes with at most two spans per series',
    layout: 'structure-of-arrays price and volume slabs backed by one Float64 arena',
    ringBufferRule: 'a batch may wrap once; the kernel splits into a tail span and a head span',
    hotPathRequirements: Object.freeze([
      'Float64Array batch sources',
      'one predictable copy loop per contiguous span',
      'monomorphic series metadata access',
      'no per-tick allocations',
    ]),
  }),
  writeWindowFeatures: Object.freeze({
    name: 'writeWindowFeatures',
    shape: 'rolling-window reduction over one or two contiguous spans',
    layout: 'contiguous price and volume slabs plus fixed-size Float64 output vectors',
    ringBufferRule: 'the read window may wrap once; the kernel reduces each contiguous span in order',
    hotPathRequirements: Object.freeze([
      'single feature vector shape',
      'numeric reduction only',
      'no polymorphic record access',
      'branch-light span handling',
    ]),
  }),
  evaluateBatch: Object.freeze({
    name: 'evaluateBatch',
    shape: 'batched feature extraction with one output vector per series/window pair',
    layout: 'contiguous Float64Array inputs and outputs with fixed FEATURE_COUNT stride',
    ringBufferRule: 'each row delegates to the ring-buffer-safe single-series reducer',
    hotPathRequirements: Object.freeze([
      'contiguous batch inputs',
      'fixed stride output writes',
      'monomorphic loop body',
      'no object-shaped batch records',
    ]),
  }),
});

const WASM_TOOLCHAIN_STRATEGY = Object.freeze({
  preferredPath: Object.freeze({
    toolchain: 'Rust stable + cargo targeting wasm32-unknown-unknown',
    runtime: 'raw WebAssembly.instantiate / instantiateStreaming with a thin JS adapter',
    boundary: 'two-hop JS boundary: orchestration JS -> WASM numeric kernel -> Float64Array outputs',
    simdGate: '-C target-feature=+simd128',
    precisionGate: 'Float64Array memory with f64x2 SIMD lanes',
    rationale: Object.freeze([
      'f64x2 preserves double-precision semantics end-to-end instead of widening or truncating to f32',
      'the same wasm artifact can run in browsers, Node, and standalone runtimes that support WebAssembly',
      'JS stays a thin transport/orchestration layer and never owns hot-path object records',
    ]),
  }),
  buildGates: Object.freeze({
    target: 'wasm32-unknown-unknown',
    rustcFlags: Object.freeze(['-C opt-level=3', '-C target-feature=+simd128']),
    featureGates: Object.freeze([
      'target_feature = "simd128"',
      'compiled f64x2 SIMD code paths',
      'Float64Array-backed linear memory',
    ]),
  }),
  runtimePolicy: Object.freeze({
    preferredArtifact: 'double_vector_boundary.simd.wasm',
    scalarArtifact: 'double_vector_boundary.scalar.wasm',
    fallbackArtifact: 'wasm/double_vector_boundary.js',
    supportedHosts: Object.freeze(['browser', 'node', 'wasmtime', 'wasmer']),
  }),
  fallbackOrder: Object.freeze(['wasm-simd', 'wasm-scalar', 'js-float64']),
  fallbackBehavior: Object.freeze({
    noSimd: 'load the scalar wasm artifact compiled from the same Rust sources with target-feature=simd128 disabled',
    noWasm: 'use the existing JS Float64 kernel and keep the adapter shape unchanged',
  }),
});

function normalizeWasmCapabilities(capabilities) {
  const input = capabilities && typeof capabilities === 'object' ? capabilities : {};
  const hasWasm = typeof input.hasWasm === 'boolean'
    ? input.hasWasm
    : Boolean(typeof WebAssembly === 'object' && WebAssembly !== null);
  const simd128 = typeof input.simd128 === 'boolean' ? input.simd128 : false;
  const float64 = typeof input.float64 === 'boolean'
    ? input.float64
    : Boolean(typeof Float64Array === 'function');
  const runtimeFamily = typeof input.runtimeFamily === 'string'
    ? input.runtimeFamily
    : (typeof process === 'object' && process !== null && process.versions && process.versions.node
      ? 'node'
      : (typeof document === 'object' && document !== null ? 'browser' : 'unknown'));

  return Object.freeze({
    hasWasm,
    simd128,
    float64,
    runtimeFamily,
  });
}

function selectWasmRuntimeStrategy(capabilities) {
  const caps = normalizeWasmCapabilities(capabilities);

  if (!caps.float64) {
    return Object.freeze({
      mode: 'unsupported',
      precision: 'unavailable',
      twoHopBoundary: true,
      reason: 'Float64Array support is required to preserve 64-bit precision',
      fallback: 'none',
      capabilities: caps,
    });
  }

  if (!caps.hasWasm) {
    return Object.freeze({
      mode: 'js-float64',
      toolchain: 'existing JS Float64 kernel',
      runtime: 'JavaScript only',
      precision: 'Float64Array',
      twoHopBoundary: true,
      rationale: 'No WebAssembly runtime is available, so the adapter must remain on the JS Float64 path',
      fallback: 'none',
      capabilities: caps,
    });
  }

  if (caps.simd128) {
    return Object.freeze({
      mode: 'wasm-simd',
      toolchain: WASM_TOOLCHAIN_STRATEGY.preferredPath.toolchain,
      runtime: WASM_TOOLCHAIN_STRATEGY.preferredPath.runtime,
      precision: 'Float64Array + f64x2',
      twoHopBoundary: true,
      buildGates: WASM_TOOLCHAIN_STRATEGY.buildGates,
      rationale: WASM_TOOLCHAIN_STRATEGY.preferredPath.rationale,
      fallback: WASM_TOOLCHAIN_STRATEGY.fallbackBehavior.noSimd,
      capabilities: caps,
    });
  }

  return Object.freeze({
    mode: 'wasm-scalar',
    toolchain: WASM_TOOLCHAIN_STRATEGY.preferredPath.toolchain,
    runtime: WASM_TOOLCHAIN_STRATEGY.preferredPath.runtime,
    precision: 'Float64Array',
    twoHopBoundary: true,
    buildGates: Object.freeze({
      target: WASM_TOOLCHAIN_STRATEGY.buildGates.target,
      rustcFlags: Object.freeze(['-C opt-level=3']),
      featureGates: Object.freeze(['Float64Array-backed linear memory']),
    }),
    rationale: 'WebAssembly is available, but SIMD is not; use the scalar wasm artifact so the host API stays stable',
    fallback: WASM_TOOLCHAIN_STRATEGY.fallbackBehavior.noWasm,
    capabilities: caps,
  });
}

function createThinAdapter(maxSeries, samplesPerSeries, arena) {
  const kernel = new DoubleVectorKernel(maxSeries, samplesPerSeries, arena);

  return Object.freeze({
    kernel,
    ingest(seriesId, price, volume) {
      return kernel.ingestTick(seriesId, price, volume);
    },
    ingestBatch(seriesId, prices, volumes, priceOffset = 0, volumeOffset = 0, count = prices.length - priceOffset) {
      return kernel.ingestBatch(seriesId, prices, volumes, priceOffset, volumeOffset, count);
    },
    evaluate(seriesId, windowSize = kernel.samplesPerSeries) {
      const out = kernel.getOutputView(seriesId);
      kernel.writeWindowFeatures(seriesId, windowSize, out, 0);
      return out;
    },
    evaluateInto(seriesId, windowSize, out, outOffset = 0) {
      return kernel.writeWindowFeatures(seriesId, windowSize, out, outOffset);
    },
    evaluateBatch(seriesIds, windowSizes, out, outOffset = 0, count = seriesIds.length) {
      return kernel.evaluateBatch(seriesIds, windowSizes, out, outOffset, count);
    },
    get memory() {
      return kernel.arena.f64;
    },
  });
}

module.exports = Object.freeze({
  FLOAT64_BYTES,
  CONTROL_SLOTS,
  META_SLOTS_PER_SERIES,
  FEATURE_COUNT,
  FEATURE_NAMES,
  KERNEL_CONTRACTS,
  WASM_TOOLCHAIN_STRATEGY,
  VALIDATION_PLAN,
  normalizeWasmCapabilities,
  selectWasmRuntimeStrategy,
  computeLayout,
  createArena,
  DoubleVectorKernel,
  createThinAdapter,
});
