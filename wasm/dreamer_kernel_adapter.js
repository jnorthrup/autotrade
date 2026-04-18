'use strict';

const {
  ABI_VERSION,
  ALIGNMENT_BYTES,
  DREAMER_BUFFER_LAYOUT,
  DREAMER_ERROR_CODES,
  FEATURE_COUNT,
  RESULT_LAYOUTS,
  RUNTIME_MODES,
  DREAMER_WASM_BOUNDARY,
  createRuntimePlan,
  normalizeCapabilities,
} = require('./dreamer_kernel_contract.js');
const {
  DreamerKernelError,
  createKernelError,
  createJsNumericCoreBackend,
  createWasmNumericCoreBackend,
  ensureReadableFloat64,
  ensureReadableUint32,
  ensureWritableFloat64,
} = require('./dreamer_numeric_core.js');

function asPositiveInteger(value, label) {
  if (!Number.isFinite(value) || Math.trunc(value) !== value) {
    throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} must be a finite integer`, { label, value });
  }
  if (value < 1) {
    throw createKernelError(DREAMER_ERROR_CODES.RANGE, `${label} must be >= 1`, { label, value });
  }
  return value;
}

function asNonNegativeInteger(value, label) {
  if (!Number.isFinite(value) || Math.trunc(value) !== value) {
    throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} must be a finite integer`, { label, value });
  }
  if (value < 0) {
    throw createKernelError(DREAMER_ERROR_CODES.RANGE, `${label} must be >= 0`, { label, value });
  }
  return value;
}

function asFiniteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} must be a finite number`, { label, value });
  }
  return value;
}

function alignUp(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function isTypedArray(value) {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function toFloat64Array(value, label) {
  if (value instanceof Float64Array) {
    return value;
  }
  if (Array.isArray(value) || isTypedArray(value)) {
    const out = new Float64Array(value.length);
    for (let i = 0; i < value.length; i += 1) {
      out[i] = asFiniteNumber(Number(value[i]), `${label}[${i}]`);
    }
    return out;
  }
  throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} must be a Float64Array or array-like of finite numbers`, { label });
}

function toUint32Array(value, label) {
  if (value instanceof Uint32Array) {
    return value;
  }
  if (Array.isArray(value) || isTypedArray(value)) {
    const out = new Uint32Array(value.length);
    for (let i = 0; i < value.length; i += 1) {
      const item = asNonNegativeInteger(Number(value[i]), `${label}[${i}]`);
      out[i] = item;
    }
    return out;
  }
  throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} must be a Uint32Array or array-like of non-negative integers`, { label });
}

function isRecoverableBackendError(error) {
  return Boolean(error && (
    error.code === DREAMER_ERROR_CODES.BACKEND_UNAVAILABLE
    || error.code === DREAMER_ERROR_CODES.ABI_MISMATCH
    || error.code === DREAMER_ERROR_CODES.WASM_TRAP
    || error.code === DREAMER_ERROR_CODES.SCRATCH_CAPACITY
  ));
}

class ScratchAllocator {
  constructor(memory, baseByteOffset, byteLength) {
    this.memory = memory;
    this.baseByteOffset = baseByteOffset;
    this.byteLength = byteLength;
    this.cursor = 0;
  }

  reset() {
    this.cursor = 0;
  }

  allocBytes(byteLength, alignment = 8) {
    const aligned = alignUp(this.cursor, alignment);
    if (aligned + byteLength > this.byteLength) {
      throw createKernelError(
        DREAMER_ERROR_CODES.SCRATCH_CAPACITY,
        `scratch buffer needs ${byteLength} additional bytes but only ${this.byteLength - aligned} remain`,
        { requestedBytes: byteLength, remainingBytes: this.byteLength - aligned },
      );
    }
    this.cursor = aligned + byteLength;
    return this.baseByteOffset + aligned;
  }

  allocFloat64(length) {
    const ptr = this.allocBytes(length * Float64Array.BYTES_PER_ELEMENT, Float64Array.BYTES_PER_ELEMENT);
    return {
      ptr,
      view: new Float64Array(this.memory.buffer, ptr, length),
    };
  }

  writeFloat64(source, offset = 0, count = source.length - offset) {
    const allocation = this.allocFloat64(count);
    allocation.view.set(source.subarray(offset, offset + count));
    return allocation;
  }

  allocUint32(length) {
    const ptr = this.allocBytes(length * Uint32Array.BYTES_PER_ELEMENT, Uint32Array.BYTES_PER_ELEMENT);
    return {
      ptr,
      view: new Uint32Array(this.memory.buffer, ptr, length),
    };
  }

  writeUint32(source, offset = 0, count = source.length - offset) {
    const allocation = this.allocUint32(count);
    allocation.view.set(source.subarray(offset, offset + count));
    return allocation;
  }
}

class DreamerKernelAdapter {
  constructor(options = {}) {
    this.layout = DREAMER_BUFFER_LAYOUT;
    this.boundary = DREAMER_WASM_BOUNDARY;
    this.maxSeries = asPositiveInteger(options.maxSeries ?? 256, 'maxSeries');
    this.samplesPerSeries = asPositiveInteger(options.samplesPerSeries ?? 1024, 'samplesPerSeries');
    this.offsets = this.layout.computeOffsets(this.maxSeries, this.samplesPerSeries);
    this.totalSlots = this.layout.computeTotalSlots(this.maxSeries, this.samplesPerSeries);
    this.requiredBytes = this.layout.computeBytes(this.maxSeries, this.samplesPerSeries);
    this.alignment = ALIGNMENT_BYTES;
    this.scratchFloat64Slots = asPositiveInteger(
      options.scratchFloat64Slots ?? Math.max(4096, this.maxSeries * FEATURE_COUNT * 2, this.samplesPerSeries * 4),
      'scratchFloat64Slots',
    );
    this.scratchBytes = this.scratchFloat64Slots * Float64Array.BYTES_PER_ELEMENT;
    this.scratchByteOffset = alignUp(this.requiredBytes, ALIGNMENT_BYTES);
    this.totalBytes = this.scratchByteOffset + this.scratchBytes;
    this.arenaPtr = 0;
    this.memory = null;
    this.u8View = null;
    this.f64View = null;
    this.arenaF64 = null;
    this.scratch = null;

    this.capabilities = normalizeCapabilities(options.capabilities);
    this.allowNumericFallback = options.allowNumericFallback !== false;
    this.lastFallbackReason = null;
    this.lastBackendError = null;
    this.compilePromise = null;

    this._batchOutput = new Float64Array(0);
    this._portfolioAggregate = new Float64Array(RESULT_LAYOUTS.PORTFOLIO.count);
    this._portfolioDeviations = new Float64Array(0);
    this._portfolioHarvest = new Float64Array(0);
    this._portfolioRebalance = new Float64Array(0);
    this._defectResult = new Float64Array(RESULT_LAYOUTS.DEFECT_SCAN.count);
    this._regimeResult = new Float64Array(RESULT_LAYOUTS.REGIME.count);

    this.jsBackend = options.jsBackend || createJsNumericCoreBackend();
    this.wasmBackend = null;
    this.customBackend = options.backend || null;

    this._bindMemory(options.memory || new WebAssembly.Memory({
      initial: this._requiredPagesFor(this.totalBytes),
      maximum: this._requiredPagesFor(this.totalBytes),
    }));
    this._selectInitialBackend(options.preferredMode);

    if (options.wasmExports || options.wasmInstance || options.wasmModule) {
      this.attachWASM(options.wasmExports || options.wasmInstance || options.wasmModule, options.preferredMode);
    } else if (options.wasmBytes) {
      this.compilePromise = this.compileAndAttach(options.wasmBytes, options.preferredMode);
    }
  }

  _requiredPagesFor(byteLength) {
    return Math.ceil(byteLength / 65536);
  }

  _captureArenaBytes() {
    if (!this.memory || !this.u8View) {
      return null;
    }
    return new Uint8Array(this.memory.buffer, this.arenaPtr, this.requiredBytes).slice();
  }

  _bindMemory(memory, arenaBytes = null) {
    if (!(memory instanceof WebAssembly.Memory)) {
      throw createKernelError(DREAMER_ERROR_CODES.TYPE, 'memory must be a WebAssembly.Memory instance');
    }

    const requiredPages = this._requiredPagesFor(this.totalBytes);
    const currentPages = this._requiredPagesFor(memory.buffer.byteLength);
    if (currentPages < requiredPages) {
      memory.grow(requiredPages - currentPages);
    }

    this.memory = memory;
    this.u8View = new Uint8Array(memory.buffer);
    this.f64View = new Float64Array(memory.buffer, this.arenaPtr, this.totalSlots);
    this.arenaF64 = this.f64View;
    this.scratch = new ScratchAllocator(memory, this.scratchByteOffset, this.scratchBytes);

    if (arenaBytes) {
      this.u8View.set(arenaBytes, this.arenaPtr);
    } else {
      this._initializeArena();
    }
  }

  _adoptBackendMemory(memory) {
    if (!(memory instanceof WebAssembly.Memory)) {
      return false;
    }
    const arenaBytes = this._captureArenaBytes();
    this._bindMemory(memory, arenaBytes);
    return true;
  }

  _initializeArena() {
    this.f64View.fill(0);
    this.f64View[this.layout.CONTROL.maxSeries] = this.maxSeries;
    this.f64View[this.layout.CONTROL.samplesPerSeries] = this.samplesPerSeries;
    this.f64View[this.layout.CONTROL.featureCount] = FEATURE_COUNT;
    this.f64View[this.layout.CONTROL.lastSeriesId] = -1;
    this.f64View[this.layout.CONTROL.lastPrice] = Number.NaN;
    this.f64View[this.layout.CONTROL.lastVolume] = Number.NaN;
  }

  _selectInitialBackend(preferredMode) {
    this.runtimePlan = createRuntimePlan(this.capabilities);
    this.preferredMode = preferredMode || this.runtimePlan.mode;
    if (this.customBackend) {
      this.activeBackend = this.customBackend;
      return;
    }
    this.activeBackend = this.jsBackend;
  }

  _selectBackendAfterAttach(preferredMode) {
    this.runtimePlan = createRuntimePlan({
      ...this.capabilities,
      hasWasm: this.capabilities.hasWasm && Boolean(this.wasmBackend),
    });
    this.preferredMode = preferredMode || this.preferredMode || this.runtimePlan.mode;

    if (this.customBackend) {
      this.activeBackend = this.customBackend;
      return;
    }

    if (this.wasmBackend && (this.preferredMode === RUNTIME_MODES.WASM_SIMD || this.preferredMode === RUNTIME_MODES.WASM_SCALAR)) {
      this.activeBackend = this.wasmBackend;
      return;
    }

    if (this.wasmBackend && this.runtimePlan.mode !== RUNTIME_MODES.JS_FLOAT64_CORE) {
      this.activeBackend = this.wasmBackend;
      return;
    }

    this.activeBackend = this.jsBackend;
  }

  _buildKernelContext() {
    this.scratch.reset();
    return {
      maxSeries: this.maxSeries,
      samplesPerSeries: this.samplesPerSeries,
      offsets: this.offsets,
      arenaPtr: this.arenaPtr,
      arenaF64: this.arenaF64,
      memory: this.memory,
      scratch: this.scratch,
      pointerForView: (view, outOffset, length) => this.pointerForView(view, outOffset, length),
      pointerOrScratch: (view, outOffset, length) => this.pointerOrScratch(view, outOffset, length),
    };
  }

  pointerForView(view, outOffset, length) {
    if (!(view instanceof Float64Array)) {
      return null;
    }
    if (view.buffer !== this.memory.buffer) {
      return null;
    }
    const byteOffset = view.byteOffset + (outOffset * Float64Array.BYTES_PER_ELEMENT);
    const available = view.length - outOffset;
    return available >= length ? byteOffset : null;
  }

  pointerOrScratch(view, outOffset, length) {
    const directPtr = this.pointerForView(view, outOffset, length);
    if (directPtr !== null) {
      return {
        ptr: directPtr,
        view: new Float64Array(this.memory.buffer, directPtr, length),
        copyBack() {},
      };
    }

    const allocation = this.scratch.allocFloat64(length);
    return {
      ptr: allocation.ptr,
      view: allocation.view,
      copyBack: () => {
        view.set(allocation.view, outOffset);
      },
    };
  }

  _normalizeSeriesId(seriesId) {
    const sid = asNonNegativeInteger(seriesId, 'seriesId');
    if (sid >= this.maxSeries) {
      throw createKernelError(DREAMER_ERROR_CODES.RANGE, `seriesId must be in [0, ${this.maxSeries - 1}]`, { seriesId: sid, maxSeries: this.maxSeries });
    }
    return sid;
  }

  _normalizeBatchCount(count, fallbackLength) {
    return asNonNegativeInteger(count ?? fallbackLength, 'count');
  }

  _normalizePortfolioParams(params = {}) {
    return Object.freeze({
      cashBalance: asFiniteNumber(params.cashBalance ?? 0, 'cashBalance'),
      harvestTrigger: asFiniteNumber(params.harvestTrigger ?? 0.05, 'harvestTrigger'),
      rebalanceTrigger: asFiniteNumber(params.rebalanceTrigger ?? 0.05, 'rebalanceTrigger'),
      cpTriggerAssetPercent: asFiniteNumber(params.cpTriggerAssetPercent ?? 0.5, 'cpTriggerAssetPercent'),
      cpTriggerMinNegativeDev: asFiniteNumber(params.cpTriggerMinNegativeDev ?? -0.05, 'cpTriggerMinNegativeDev'),
    });
  }

  _ensureBatchOutput(count) {
    const required = count * FEATURE_COUNT;
    if (this._batchOutput.length < required) {
      this._batchOutput = new Float64Array(required);
    }
    return this._batchOutput.subarray(0, required);
  }

  _ensurePortfolioScratch(assetCount) {
    if (this._portfolioDeviations.length < assetCount) {
      this._portfolioDeviations = new Float64Array(assetCount);
      this._portfolioHarvest = new Float64Array(assetCount);
      this._portfolioRebalance = new Float64Array(assetCount);
    }
    return {
      aggregate: this._portfolioAggregate,
      deviations: this._portfolioDeviations.subarray(0, assetCount),
      harvest: this._portfolioHarvest.subarray(0, assetCount),
      rebalance: this._portfolioRebalance.subarray(0, assetCount),
    };
  }

  _invoke(operation, args) {
    const backend = this.activeBackend || this.jsBackend;
    const context = this._buildKernelContext();
    try {
      return backend[operation](context, args);
    } catch (error) {
      this.lastBackendError = error;
      if (backend !== this.jsBackend && this.allowNumericFallback && isRecoverableBackendError(error)) {
        this.lastFallbackReason = `${backend.kind || 'backend'}:${error.code || error.message}`;
        this.activeBackend = this.jsBackend;
        return this.jsBackend[operation](context, args);
      }
      if (error && error.name === 'DreamerKernelError') {
        throw error;
      }
      throw createKernelError(
        DREAMER_ERROR_CODES.BACKEND_UNAVAILABLE,
        `failed to execute ${operation}: ${error.message}`,
        { operation, cause: error },
      );
    }
  }

  getMetaView(seriesId) {
    const sid = this._normalizeSeriesId(seriesId);
    const start = this.offsets.metaStart + (sid * this.layout.SERIES_META.slotsPerSeries);
    return this.f64View.subarray(start, start + this.layout.SERIES_META.slotsPerSeries);
  }

  getPriceView(seriesId) {
    const sid = this._normalizeSeriesId(seriesId);
    const start = this.offsets.pricesStart + (sid * this.samplesPerSeries);
    return this.f64View.subarray(start, start + this.samplesPerSeries);
  }

  getVolumeView(seriesId) {
    const sid = this._normalizeSeriesId(seriesId);
    const start = this.offsets.volumesStart + (sid * this.samplesPerSeries);
    return this.f64View.subarray(start, start + this.samplesPerSeries);
  }

  getFeatureView(seriesId) {
    const sid = this._normalizeSeriesId(seriesId);
    const start = this.offsets.featuresStart + (sid * FEATURE_COUNT);
    return this.f64View.subarray(start, start + FEATURE_COUNT);
  }

  getWriteIndex(seriesId) {
    return this.getMetaView(seriesId)[this.layout.SERIES_META.writeIndex] | 0;
  }

  getSampleCount(seriesId) {
    return this.getMetaView(seriesId)[this.layout.SERIES_META.sampleCount] | 0;
  }

  ingestTick(seriesId, price, volume) {
    return this._invoke('ingestTick', {
      seriesId: this._normalizeSeriesId(seriesId),
      price: asFiniteNumber(price, 'price'),
      volume: asFiniteNumber(volume, 'volume'),
    });
  }

  ingestBatch(seriesId, prices, volumes, priceOffset = 0, volumeOffset = 0, count = prices.length - priceOffset) {
    ensureReadableFloat64(prices, 'prices');
    ensureReadableFloat64(volumes, 'volumes');
    const sid = this._normalizeSeriesId(seriesId);
    const sourcePriceOffset = asNonNegativeInteger(priceOffset, 'priceOffset');
    const sourceVolumeOffset = asNonNegativeInteger(volumeOffset, 'volumeOffset');
    const batchCount = this._normalizeBatchCount(count, prices.length - sourcePriceOffset);

    if (sourcePriceOffset + batchCount > prices.length) {
      throw createKernelError(DREAMER_ERROR_CODES.RANGE, 'prices does not contain enough samples for the requested batch', { priceOffset: sourcePriceOffset, batchCount, pricesLength: prices.length });
    }
    if (sourceVolumeOffset + batchCount > volumes.length) {
      throw createKernelError(DREAMER_ERROR_CODES.RANGE, 'volumes does not contain enough samples for the requested batch', { volumeOffset: sourceVolumeOffset, batchCount, volumesLength: volumes.length });
    }
    if (batchCount > this.samplesPerSeries) {
      throw createKernelError(DREAMER_ERROR_CODES.RANGE, `count must be <= samplesPerSeries (${this.samplesPerSeries}) for the stable one-wrap batch contract`, { batchCount, samplesPerSeries: this.samplesPerSeries });
    }
    if (batchCount === 0) {
      return this.getSampleCount(sid);
    }

    return this._invoke('ingestBatch', {
      seriesId: sid,
      prices,
      volumes,
      priceOffset: sourcePriceOffset,
      volumeOffset: sourceVolumeOffset,
      count: batchCount,
    });
  }

  computeFeatures(seriesId, windowSize) {
    return this._invoke('computeFeatures', {
      seriesId: this._normalizeSeriesId(seriesId),
      windowSize: asPositiveInteger(windowSize, 'windowSize'),
    });
  }

  computeFeaturesInto(seriesId, windowSize, out, outOffset = 0) {
    ensureWritableFloat64(out, outOffset, FEATURE_COUNT, 'feature output');
    return this._invoke('computeFeaturesInto', {
      seriesId: this._normalizeSeriesId(seriesId),
      windowSize: asPositiveInteger(windowSize, 'windowSize'),
      out,
      outOffset: asNonNegativeInteger(outOffset, 'outOffset'),
    });
  }

  computeFeaturesBatch(seriesIds, windowSizes, count = seriesIds.length) {
    const ids = toUint32Array(seriesIds, 'seriesIds');
    const wins = toUint32Array(windowSizes, 'windowSizes');
    const batchCount = this._normalizeBatchCount(count, ids.length);
    ensureReadableUint32(ids, 'seriesIds', batchCount);
    ensureReadableUint32(wins, 'windowSizes', batchCount);
    return this._invoke('computeFeaturesBatch', {
      seriesIds: ids,
      windowSizes: wins,
      count: batchCount,
    });
  }

  computeFeaturesBatchInto(seriesIds, windowSizes, out, outOffset = 0, count = seriesIds.length) {
    const ids = toUint32Array(seriesIds, 'seriesIds');
    const wins = toUint32Array(windowSizes, 'windowSizes');
    const batchCount = this._normalizeBatchCount(count, ids.length);
    ensureReadableUint32(ids, 'seriesIds', batchCount);
    ensureReadableUint32(wins, 'windowSizes', batchCount);
    ensureWritableFloat64(out, outOffset, batchCount * FEATURE_COUNT, 'batch feature output');
    return this._invoke('computeFeaturesBatchInto', {
      seriesIds: ids,
      windowSizes: wins,
      count: batchCount,
      out,
      outOffset: asNonNegativeInteger(outOffset, 'outOffset'),
    });
  }

  computeFeaturesBatchFixed(seriesIds, windowSize, count = seriesIds.length) {
    const ids = toUint32Array(seriesIds, 'seriesIds');
    const batchCount = this._normalizeBatchCount(count, ids.length);
    ensureReadableUint32(ids, 'seriesIds', batchCount);
    return this._invoke('computeFeaturesBatchFixed', {
      seriesIds: ids,
      windowSize: asPositiveInteger(windowSize, 'windowSize'),
      count: batchCount,
    });
  }

  computeFeaturesBatchFixedInto(seriesIds, windowSize, out, outOffset = 0, count = seriesIds.length) {
    const ids = toUint32Array(seriesIds, 'seriesIds');
    const batchCount = this._normalizeBatchCount(count, ids.length);
    ensureReadableUint32(ids, 'seriesIds', batchCount);
    ensureWritableFloat64(out, outOffset, batchCount * FEATURE_COUNT, 'fixed-window batch feature output');
    return this._invoke('computeFeaturesBatchFixedInto', {
      seriesIds: ids,
      windowSize: asPositiveInteger(windowSize, 'windowSize'),
      count: batchCount,
      out,
      outOffset: asNonNegativeInteger(outOffset, 'outOffset'),
    });
  }

  evaluateBatch(seriesIds, windowSizes, out, outOffset = 0, count = seriesIds.length) {
    return this.computeFeaturesBatchInto(seriesIds, windowSizes, out, outOffset, count);
  }

  computePortfolioInto(values, baselines, params, aggregateOut, deviationsOut, harvestOut, rebalanceOut) {
    const valueSeries = toFloat64Array(values, 'values');
    const baselineSeries = toFloat64Array(baselines, 'baselines');
    if (valueSeries.length !== baselineSeries.length) {
      throw createKernelError(DREAMER_ERROR_CODES.RANGE, 'values and baselines must have the same length', { valuesLength: valueSeries.length, baselinesLength: baselineSeries.length });
    }
    const normalizedParams = this._normalizePortfolioParams(params);
    ensureWritableFloat64(aggregateOut, 0, RESULT_LAYOUTS.PORTFOLIO.count, 'portfolio aggregate output');
    ensureWritableFloat64(deviationsOut, 0, valueSeries.length, 'portfolio deviations output');
    ensureWritableFloat64(harvestOut, 0, valueSeries.length, 'portfolio harvest output');
    ensureWritableFloat64(rebalanceOut, 0, valueSeries.length, 'portfolio rebalance output');

    return this._invoke('computePortfolioInto', {
      values: valueSeries,
      baselines: baselineSeries,
      params: normalizedParams,
      aggregateOut,
      deviationsOut,
      harvestOut,
      rebalanceOut,
    });
  }

  computePortfolio(values, baselines, params = {}) {
    const valueSeries = toFloat64Array(values, 'values');
    const baselineSeries = toFloat64Array(baselines, 'baselines');
    const out = this._ensurePortfolioScratch(valueSeries.length);
    this.computePortfolioInto(valueSeries, baselineSeries, params, out.aggregate, out.deviations, out.harvest, out.rebalance);
    return {
      aggregate: out.aggregate,
      deviations: out.deviations,
      harvestCandidates: out.harvest,
      rebalanceCandidates: out.rebalance,
      deviationPercent: out.aggregate[RESULT_LAYOUTS.PORTFOLIO.indices.DEVIATION_PERCENT],
      crashActive: out.aggregate[RESULT_LAYOUTS.PORTFOLIO.indices.CRASH_ACTIVE],
      decliningCount: out.aggregate[RESULT_LAYOUTS.PORTFOLIO.indices.DECLINING_COUNT],
      managedBaseline: out.aggregate[RESULT_LAYOUTS.PORTFOLIO.indices.MANAGED_BASELINE],
      baselineDiff: out.aggregate[RESULT_LAYOUTS.PORTFOLIO.indices.BASELINE_DIFF],
    };
  }

  scanDefectsInto(prices, rebalanceTrigger, crashThreshold = 0.01, out = this._defectResult) {
    const priceSeries = toFloat64Array(prices, 'prices');
    ensureWritableFloat64(out, 0, RESULT_LAYOUTS.DEFECT_SCAN.count, 'defect output');
    return this._invoke('scanDefectsInto', {
      prices: priceSeries,
      rebalanceTrigger: asFiniteNumber(rebalanceTrigger, 'rebalanceTrigger'),
      crashThreshold: asFiniteNumber(crashThreshold, 'crashThreshold'),
      out,
    });
  }

  scanDefects(prices, rebalanceTrigger, crashThreshold = 0.01) {
    const out = this.scanDefectsInto(prices, rebalanceTrigger, crashThreshold, this._defectResult);
    return {
      out,
      isDefective: Boolean(out[RESULT_LAYOUTS.DEFECT_SCAN.indices.IS_DEFECTIVE]),
      maxDrawdown: out[RESULT_LAYOUTS.DEFECT_SCAN.indices.MAX_DRAWDOWN],
      triggerHits: out[RESULT_LAYOUTS.DEFECT_SCAN.indices.TRIGGER_HITS],
    };
  }

  computeRegimeInto(history, currentPrice, startPrice, out = this._regimeResult) {
    const historySeries = toFloat64Array(history, 'history');
    ensureWritableFloat64(out, 0, RESULT_LAYOUTS.REGIME.count, 'regime output');
    return this._invoke('computeRegimeInto', {
      history: historySeries,
      currentPrice: asFiniteNumber(currentPrice, 'currentPrice'),
      startPrice: asFiniteNumber(startPrice, 'startPrice'),
      out,
    });
  }

  computeRegime(history, currentPrice, startPrice) {
    const out = this.computeRegimeInto(history, currentPrice, startPrice, this._regimeResult);
    return {
      out,
      regimeCode: out[RESULT_LAYOUTS.REGIME.indices.REGIME_CODE],
      roi: out[RESULT_LAYOUTS.REGIME.indices.ROI],
      volatility: out[RESULT_LAYOUTS.REGIME.indices.VOLATILITY],
      mean: out[RESULT_LAYOUTS.REGIME.indices.MEAN],
    };
  }

  attachWASM(moduleOrInstanceOrExports, preferredMode = null) {
    let exports = null;
    if (moduleOrInstanceOrExports instanceof WebAssembly.Instance) {
      exports = moduleOrInstanceOrExports.exports;
    } else if (moduleOrInstanceOrExports instanceof WebAssembly.Module) {
      const instance = new WebAssembly.Instance(moduleOrInstanceOrExports, {
        env: {
          memory: this.memory,
          __memory_base: this.arenaPtr,
        },
      });
      exports = instance.exports;
    } else if (moduleOrInstanceOrExports && typeof moduleOrInstanceOrExports === 'object' && moduleOrInstanceOrExports.exports) {
      exports = moduleOrInstanceOrExports.exports;
    } else if (moduleOrInstanceOrExports && typeof moduleOrInstanceOrExports === 'object') {
      exports = moduleOrInstanceOrExports;
    }

    if (!exports) {
      throw createKernelError(DREAMER_ERROR_CODES.TYPE, 'attachWASM requires a WebAssembly.Module, WebAssembly.Instance, or exports object');
    }

    this._adoptBackendMemory(exports.memory);

    const mode = preferredMode || this.preferredMode || (this.capabilities.simd128 ? RUNTIME_MODES.WASM_SIMD : RUNTIME_MODES.WASM_SCALAR);
    this.wasmBackend = createWasmNumericCoreBackend(exports, mode);
    this._selectBackendAfterAttach(mode);
    return this;
  }

  async compileAndAttach(bytes, preferredMode = null) {
    if (!(bytes instanceof ArrayBuffer || ArrayBuffer.isView(bytes))) {
      throw createKernelError(DREAMER_ERROR_CODES.TYPE, 'compileAndAttach requires wasm bytes as an ArrayBuffer or typed array');
    }
    const source = bytes instanceof ArrayBuffer ? bytes : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const module = await WebAssembly.compile(source);
    this.attachWASM(module, preferredMode);
    return this;
  }

  getABIVersion() {
    const active = this.activeBackend || this.jsBackend;
    return typeof active.getAbiVersion === 'function' ? active.getAbiVersion() : ABI_VERSION;
  }

  isReady() {
    return Boolean(this.activeBackend);
  }

  getStats() {
    return {
      maxSeries: this.maxSeries,
      samplesPerSeries: this.samplesPerSeries,
      totalSlots: this.totalSlots,
      requiredBytes: this.requiredBytes,
      scratchBytes: this.scratchBytes,
      totalBytes: this.totalBytes,
      alignment: this.alignment,
      arenaPtr: this.arenaPtr,
      memoryByteLength: this.memory.buffer.byteLength,
      memorySource: this.wasmBackend && this.wasmBackend.exports && this.wasmBackend.exports.memory === this.memory
        ? 'wasm-export'
        : 'adapter-local',
      offsets: this.offsets,
      abiVersion: this.getABIVersion(),
      runtimePlan: this.runtimePlan,
      activeBackend: this.activeBackend ? this.activeBackend.kind : null,
      fallbackReason: this.lastFallbackReason,
      wasmAttached: Boolean(this.wasmBackend),
      wasmReady: Boolean(this.wasmBackend && this.activeBackend === this.wasmBackend),
      signedOffHotPaths: this.boundary.hotPathScope.filter((entry) => entry.signedOff).map((entry) => entry.id),
    };
  }
}

module.exports = {
  DreamerKernelAdapter,
  DreamerKernelError,
  DREAMER_BUFFER_LAYOUT: DREAMER_BUFFER_LAYOUT,
  DREAMER_WASM_BOUNDARY,
  RUNTIME_MODES,
};
