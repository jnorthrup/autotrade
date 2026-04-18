'use strict';

const {
  ABI_VERSION,
  DREAMER_BUFFER_LAYOUT,
  DREAMER_ERROR_CODES,
  FEATURE_COUNT,
  FEATURE_INDICES,
  REGIME_CODES,
  RESULT_LAYOUTS,
  RUNTIME_MODES,
} = require('./dreamer_kernel_contract.js');

class DreamerKernelError extends Error {
  constructor(code, message, details = {}) {
    super(message);
    this.name = 'DreamerKernelError';
    this.code = code;
    Object.assign(this, details);
  }
}

function createKernelError(code, message, details = {}) {
  return new DreamerKernelError(code, message, details);
}

function isFloat64Array(value) {
  return value instanceof Float64Array;
}

function isUint32Array(value) {
  return value instanceof Uint32Array;
}

function sameTypedRange(view, buffer, byteOffset, length) {
  return view.buffer === buffer && view.byteOffset === byteOffset && view.length === length;
}

function ensureWritableFloat64(out, outOffset, requiredSlots, label) {
  if (!isFloat64Array(out)) {
    throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} must be a Float64Array`, { label });
  }
  if (!Number.isFinite(outOffset) || Math.trunc(outOffset) !== outOffset || outOffset < 0) {
    throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} offset must be a non-negative integer`, { label, outOffset });
  }
  if (out.length - outOffset < requiredSlots) {
    throw createKernelError(
      DREAMER_ERROR_CODES.RANGE,
      `${label} needs ${requiredSlots} writable Float64 slots starting at offset ${outOffset}`,
      { label, outOffset, requiredSlots, available: out.length - outOffset },
    );
  }
}

function ensureReadableFloat64(input, label, minLength = 0) {
  if (!isFloat64Array(input)) {
    throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} must be a Float64Array`, { label });
  }
  if (input.length < minLength) {
    throw createKernelError(DREAMER_ERROR_CODES.RANGE, `${label} must contain at least ${minLength} elements`, { label, minLength, length: input.length });
  }
}

function ensureReadableUint32(input, label, minLength = 0) {
  if (!isUint32Array(input)) {
    throw createKernelError(DREAMER_ERROR_CODES.TYPE, `${label} must be a Uint32Array`, { label });
  }
  if (input.length < minLength) {
    throw createKernelError(DREAMER_ERROR_CODES.RANGE, `${label} must contain at least ${minLength} elements`, { label, minLength, length: input.length });
  }
}

function metaSlot(context, seriesId) {
  return context.offsets.metaStart + (seriesId * DREAMER_BUFFER_LAYOUT.SERIES_META.slotsPerSeries);
}

function priceSlot(context, seriesId) {
  return context.offsets.pricesStart + (seriesId * context.samplesPerSeries);
}

function volumeSlot(context, seriesId) {
  return context.offsets.volumesStart + (seriesId * context.samplesPerSeries);
}

function featureSlot(context, seriesId) {
  return context.offsets.featuresStart + (seriesId * FEATURE_COUNT);
}

function featureView(context, seriesId) {
  const start = featureSlot(context, seriesId);
  return context.arenaF64.subarray(start, start + FEATURE_COUNT);
}

function readSeriesMeta(context, seriesId) {
  const slot = metaSlot(context, seriesId);
  return {
    writeIndex: context.arenaF64[slot] | 0,
    sampleCount: context.arenaF64[slot + 1] | 0,
  };
}

function writeSeriesMeta(context, seriesId, writeIndex, sampleCount) {
  const slot = metaSlot(context, seriesId);
  context.arenaF64[slot] = writeIndex;
  context.arenaF64[slot + 1] = sampleCount;
}

function writeFeatureVector(out, outOffset, meanPrice, variance, vwap, latestPrice, priceMomentum, meanVolume) {
  out[outOffset + FEATURE_INDICES.MEAN_PRICE] = meanPrice;
  out[outOffset + FEATURE_INDICES.PRICE_VARIANCE] = variance;
  out[outOffset + FEATURE_INDICES.VWAP] = vwap;
  out[outOffset + FEATURE_INDICES.LATEST_PRICE] = latestPrice;
  out[outOffset + FEATURE_INDICES.PRICE_MOMENTUM] = priceMomentum;
  out[outOffset + FEATURE_INDICES.MEAN_VOLUME] = meanVolume;
}

function computeFeatureVectorInto(context, seriesId, windowSize, out, outOffset) {
  ensureWritableFloat64(out, outOffset, FEATURE_COUNT, 'feature output');

  const { writeIndex, sampleCount } = readSeriesMeta(context, seriesId);
  const usableWindow = Math.min(sampleCount, windowSize);
  if (usableWindow <= 0) {
    for (let i = 0; i < FEATURE_COUNT; i += 1) {
      out[outOffset + i] = Number.NaN;
    }
    return out;
  }

  const priceBase = priceSlot(context, seriesId);
  const volumeBase = volumeSlot(context, seriesId);
  const windowStart = writeIndex >= usableWindow
    ? writeIndex - usableWindow
    : context.samplesPerSeries + writeIndex - usableWindow;
  const firstSpan = Math.min(usableWindow, context.samplesPerSeries - windowStart);
  const secondSpan = usableWindow - firstSpan;

  let sumPrice = 0;
  let sumPriceSq = 0;
  let sumVolume = 0;
  let notional = 0;
  let oldestPrice = 0;
  let latestPrice = 0;

  let slot = priceBase + windowStart;
  let price = context.arenaF64[slot];
  let volume = context.arenaF64[volumeBase + windowStart];
  oldestPrice = price;
  latestPrice = price;
  sumPrice = price;
  sumPriceSq = price * price;
  sumVolume = volume;
  notional = price * volume;

  for (let i = 1; i < firstSpan; i += 1) {
    price = context.arenaF64[slot + i];
    volume = context.arenaF64[volumeBase + windowStart + i];
    latestPrice = price;
    sumPrice += price;
    sumPriceSq += price * price;
    sumVolume += volume;
    notional += price * volume;
  }

  if (secondSpan > 0) {
    slot = priceBase;
    volume = context.arenaF64[volumeBase];
    price = context.arenaF64[slot];
    latestPrice = price;
    sumPrice += price;
    sumPriceSq += price * price;
    sumVolume += volume;
    notional += price * volume;

    for (let i = 1; i < secondSpan; i += 1) {
      price = context.arenaF64[slot + i];
      volume = context.arenaF64[volumeBase + i];
      latestPrice = price;
      sumPrice += price;
      sumPriceSq += price * price;
      sumVolume += volume;
      notional += price * volume;
    }
  }

  const count = usableWindow;
  const meanPrice = sumPrice / count;
  let variance = (sumPriceSq / count) - (meanPrice * meanPrice);
  if (variance < 0 && variance > -1e-12) {
    variance = 0;
  }
  const vwap = sumVolume === 0 ? Number.NaN : notional / sumVolume;
  const priceMomentum = latestPrice - oldestPrice;
  const meanVolume = sumVolume / count;

  writeFeatureVector(out, outOffset, meanPrice, variance, vwap, latestPrice, priceMomentum, meanVolume);
  return out;
}

function computePortfolioInto(values, baselines, params, aggregateOut, deviationsOut, harvestOut, rebalanceOut) {
  const assetCount = values.length;
  ensureReadableFloat64(values, 'values');
  ensureReadableFloat64(baselines, 'baselines', assetCount);
  ensureWritableFloat64(aggregateOut, 0, RESULT_LAYOUTS.PORTFOLIO.count, 'portfolio aggregate output');
  ensureWritableFloat64(deviationsOut, 0, assetCount, 'portfolio deviations output');
  ensureWritableFloat64(harvestOut, 0, assetCount, 'portfolio harvest output');
  ensureWritableFloat64(rebalanceOut, 0, assetCount, 'portfolio rebalance output');

  const {
    harvestTrigger = 0.05,
    rebalanceTrigger = 0.05,
    cpTriggerAssetPercent = 0.5,
    cpTriggerMinNegativeDev = -0.05,
  } = params;

  let totalBaselineDiff = 0;
  let totalManagedBaseline = 0;
  let decliningCount = 0;

  for (let i = 0; i < assetCount; i += 1) {
    const value = values[i];
    const baseline = baselines[i];
    const deviation = baseline > 0 ? (value - baseline) / baseline : 0;

    deviationsOut[i] = deviation;
    harvestOut[i] = deviation >= harvestTrigger ? 1 : 0;
    rebalanceOut[i] = deviation <= -rebalanceTrigger ? 1 : 0;

    if (baseline > 0) {
      totalBaselineDiff += (value - baseline);
      totalManagedBaseline += baseline;
      if (deviation <= cpTriggerMinNegativeDev) {
        decliningCount += 1;
      }
    }
  }

  const deviationPercent = totalManagedBaseline > 0
    ? (totalBaselineDiff / totalManagedBaseline) * 100
    : 0;
  const crashActive = totalManagedBaseline > 0 && assetCount > 0
    ? ((decliningCount / assetCount) >= cpTriggerAssetPercent ? 1 : 0)
    : 0;

  aggregateOut[RESULT_LAYOUTS.PORTFOLIO.indices.DEVIATION_PERCENT] = deviationPercent;
  aggregateOut[RESULT_LAYOUTS.PORTFOLIO.indices.CRASH_ACTIVE] = crashActive;
  aggregateOut[RESULT_LAYOUTS.PORTFOLIO.indices.DECLINING_COUNT] = decliningCount;
  aggregateOut[RESULT_LAYOUTS.PORTFOLIO.indices.MANAGED_BASELINE] = totalManagedBaseline;
  aggregateOut[RESULT_LAYOUTS.PORTFOLIO.indices.BASELINE_DIFF] = totalBaselineDiff;
  return aggregateOut;
}

function scanDefectsInto(prices, rebalanceTrigger, crashThreshold, out) {
  ensureReadableFloat64(prices, 'prices');
  ensureWritableFloat64(out, 0, RESULT_LAYOUTS.DEFECT_SCAN.count, 'defect output');

  const priceCount = prices.length;
  if (priceCount < 2) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    return out;
  }

  let maxPrice = prices[0];
  let isDefective = 0;
  let maxDrawdown = 0;
  let triggerHits = 0;

  for (let i = 0; i < priceCount; i += 1) {
    const currentPrice = prices[i];
    if (currentPrice > maxPrice) {
      maxPrice = currentPrice;
    }

    const deviation = maxPrice !== 0 ? (currentPrice - maxPrice) / maxPrice : Number.NaN;
    if (deviation < rebalanceTrigger) {
      triggerHits += 1;
      let minFuturePrice = currentPrice;
      for (let j = i + 1; j < priceCount; j += 1) {
        if (prices[j] < minFuturePrice) {
          minFuturePrice = prices[j];
        }
      }
      const subsequentDrop = currentPrice !== 0 ? (minFuturePrice - currentPrice) / currentPrice : Number.NaN;
      if (subsequentDrop < -crashThreshold) {
        isDefective = 1;
        const drawdown = -subsequentDrop;
        if (drawdown > maxDrawdown) {
          maxDrawdown = drawdown;
        }
      }
    }
  }

  out[RESULT_LAYOUTS.DEFECT_SCAN.indices.IS_DEFECTIVE] = isDefective;
  out[RESULT_LAYOUTS.DEFECT_SCAN.indices.MAX_DRAWDOWN] = maxDrawdown;
  out[RESULT_LAYOUTS.DEFECT_SCAN.indices.TRIGGER_HITS] = triggerHits;
  return out;
}

function computeRegimeInto(history, currentPrice, startPrice, out) {
  ensureReadableFloat64(history, 'history');
  ensureWritableFloat64(out, 0, RESULT_LAYOUTS.REGIME.count, 'regime output');

  if (history.length < 50 || startPrice <= 0) {
    out[RESULT_LAYOUTS.REGIME.indices.REGIME_CODE] = REGIME_CODES.UNKNOWN;
    out[RESULT_LAYOUTS.REGIME.indices.ROI] = 0;
    out[RESULT_LAYOUTS.REGIME.indices.VOLATILITY] = 0;
    out[RESULT_LAYOUTS.REGIME.indices.MEAN] = 0;
    return out;
  }

  let sum = 0;
  for (let i = 0; i < history.length; i += 1) {
    sum += history[i];
  }
  const mean = sum / history.length;

  let sumSqDiff = 0;
  for (let i = 0; i < history.length; i += 1) {
    const diff = history[i] - mean;
    sumSqDiff += diff * diff;
  }

  const variance = sumSqDiff / history.length;
  const volatility = mean > 0 ? Math.sqrt(variance) / mean : 0;
  const roi = (currentPrice - startPrice) / startPrice;

  let regimeCode = REGIME_CODES.CRAB_CHOP;
  if (roi > 0.05 && volatility > 0.02) {
    regimeCode = REGIME_CODES.BULL_RUSH;
  } else if (roi < -0.05 && volatility > 0.02) {
    regimeCode = REGIME_CODES.BEAR_CRASH;
  } else if (roi > 0.02 && volatility < 0.01) {
    regimeCode = REGIME_CODES.STEADY_GROWTH;
  } else if (volatility > 0.05) {
    regimeCode = REGIME_CODES.VOLATILE_CHOP;
  }

  out[RESULT_LAYOUTS.REGIME.indices.REGIME_CODE] = regimeCode;
  out[RESULT_LAYOUTS.REGIME.indices.ROI] = roi;
  out[RESULT_LAYOUTS.REGIME.indices.VOLATILITY] = volatility;
  out[RESULT_LAYOUTS.REGIME.indices.MEAN] = mean;
  return out;
}

class JsFloat64NumericCoreBackend {
  constructor() {
    this.kind = RUNTIME_MODES.JS_FLOAT64_CORE;
    this.isWasm = false;
  }

  getAbiVersion() {
    return ABI_VERSION;
  }

  ingestTick(context, args) {
    const { seriesId, price, volume } = args;
    const { writeIndex, sampleCount } = readSeriesMeta(context, seriesId);
    const priceBase = priceSlot(context, seriesId);
    const volumeBase = volumeSlot(context, seriesId);
    context.arenaF64[priceBase + writeIndex] = price;
    context.arenaF64[volumeBase + writeIndex] = volume;

    const nextWriteIndex = (writeIndex + 1) === context.samplesPerSeries ? 0 : (writeIndex + 1);
    const nextSampleCount = sampleCount < context.samplesPerSeries ? sampleCount + 1 : sampleCount;
    writeSeriesMeta(context, seriesId, nextWriteIndex, nextSampleCount);
    context.arenaF64[DREAMER_BUFFER_LAYOUT.CONTROL.lastSeriesId] = seriesId;
    context.arenaF64[DREAMER_BUFFER_LAYOUT.CONTROL.lastPrice] = price;
    context.arenaF64[DREAMER_BUFFER_LAYOUT.CONTROL.lastVolume] = volume;
    return nextSampleCount;
  }

  ingestBatch(context, args) {
    const {
      seriesId,
      prices,
      volumes,
      priceOffset = 0,
      volumeOffset = 0,
      count = prices.length - priceOffset,
    } = args;

    let { writeIndex, sampleCount } = readSeriesMeta(context, seriesId);
    const priceBase = priceSlot(context, seriesId);
    const volumeBase = volumeSlot(context, seriesId);

    for (let i = 0; i < count; i += 1) {
      const price = prices[priceOffset + i];
      const volume = volumes[volumeOffset + i];
      context.arenaF64[priceBase + writeIndex] = price;
      context.arenaF64[volumeBase + writeIndex] = volume;
      writeIndex += 1;
      if (writeIndex === context.samplesPerSeries) {
        writeIndex = 0;
      }
      if (sampleCount < context.samplesPerSeries) {
        sampleCount += 1;
      }
    }

    writeSeriesMeta(context, seriesId, writeIndex, sampleCount);
    context.arenaF64[DREAMER_BUFFER_LAYOUT.CONTROL.lastSeriesId] = seriesId;
    context.arenaF64[DREAMER_BUFFER_LAYOUT.CONTROL.lastPrice] = prices[priceOffset + count - 1];
    context.arenaF64[DREAMER_BUFFER_LAYOUT.CONTROL.lastVolume] = volumes[volumeOffset + count - 1];
    return sampleCount;
  }

  computeFeatures(context, args) {
    const out = featureView(context, args.seriesId);
    computeFeatureVectorInto(context, args.seriesId, args.windowSize, out, 0);
    return out;
  }

  computeFeaturesInto(context, args) {
    return computeFeatureVectorInto(context, args.seriesId, args.windowSize, args.out, args.outOffset || 0);
  }

  computeFeaturesBatch(context, args) {
    const { seriesIds, windowSizes, count = seriesIds.length } = args;
    for (let i = 0; i < count; i += 1) {
      this.computeFeatures(context, { seriesId: seriesIds[i], windowSize: windowSizes[i] });
    }
    return count;
  }

  computeFeaturesBatchInto(context, args) {
    const { seriesIds, windowSizes, count = seriesIds.length, out, outOffset = 0 } = args;
    for (let i = 0; i < count; i += 1) {
      computeFeatureVectorInto(context, seriesIds[i], windowSizes[i], out, outOffset + (i * FEATURE_COUNT));
    }
    return out;
  }

  computeFeaturesBatchFixed(context, args) {
    const { seriesIds, windowSize, count = seriesIds.length } = args;
    for (let i = 0; i < count; i += 1) {
      this.computeFeatures(context, { seriesId: seriesIds[i], windowSize });
    }
    return count;
  }

  computeFeaturesBatchFixedInto(context, args) {
    const { seriesIds, windowSize, count = seriesIds.length, out, outOffset = 0 } = args;
    for (let i = 0; i < count; i += 1) {
      computeFeatureVectorInto(context, seriesIds[i], windowSize, out, outOffset + (i * FEATURE_COUNT));
    }
    return out;
  }

  computePortfolioInto(_context, args) {
    return computePortfolioInto(
      args.values,
      args.baselines,
      args.params,
      args.aggregateOut,
      args.deviationsOut,
      args.harvestOut,
      args.rebalanceOut,
    );
  }

  scanDefectsInto(_context, args) {
    return scanDefectsInto(args.prices, args.rebalanceTrigger, args.crashThreshold, args.out);
  }

  computeRegimeInto(_context, args) {
    return computeRegimeInto(args.history, args.currentPrice, args.startPrice, args.out);
  }
}

class WasmNumericCoreBackend {
  constructor(exports, mode = RUNTIME_MODES.WASM_SCALAR) {
    this.exports = exports || null;
    this.kind = mode;
    this.isWasm = true;
  }

  getAbiVersion() {
    if (!this.exports || typeof this.exports.get_abi_version !== 'function') {
      return null;
    }
    return this.exports.get_abi_version();
  }

  requireExport(name) {
    if (!this.exports || typeof this.exports[name] !== 'function') {
      throw createKernelError(
        DREAMER_ERROR_CODES.BACKEND_UNAVAILABLE,
        `WASM backend does not export ${name}`,
        { exportName: name, backend: this.kind },
      );
    }
    return this.exports[name];
  }

  validateAbi() {
    const abiVersion = this.getAbiVersion();
    if (abiVersion !== null && abiVersion !== ABI_VERSION) {
      throw createKernelError(
        DREAMER_ERROR_CODES.ABI_MISMATCH,
        `WASM ABI mismatch: expected ${ABI_VERSION}, received ${abiVersion}`,
        { expected: ABI_VERSION, received: abiVersion, backend: this.kind },
      );
    }
  }

  runExport(name, args) {
    this.validateAbi();
    const fn = this.requireExport(name);
    try {
      return fn(...args);
    } catch (error) {
      throw createKernelError(
        DREAMER_ERROR_CODES.WASM_TRAP,
        `WASM backend trapped while executing ${name}: ${error.message}`,
        { cause: error, exportName: name, backend: this.kind },
      );
    }
  }

  ingestTick(context, args) {
    return this.runExport('ingest_tick', [
      context.arenaPtr,
      args.seriesId,
      args.price,
      args.volume,
      context.maxSeries,
      context.samplesPerSeries,
    ]);
  }

  ingestBatch(context, args) {
    const { ptr: pricesPtr } = context.scratch.writeFloat64(args.prices, args.priceOffset || 0, args.count);
    const { ptr: volumesPtr } = context.scratch.writeFloat64(args.volumes, args.volumeOffset || 0, args.count);
    return this.runExport('ingest_batch', [
      context.arenaPtr,
      args.seriesId,
      pricesPtr,
      volumesPtr,
      args.count,
      context.maxSeries,
      context.samplesPerSeries,
    ]);
  }

  computeFeatures(context, args) {
    this.runExport('compute_features', [
      context.arenaPtr,
      args.seriesId,
      args.windowSize,
      context.maxSeries,
      context.samplesPerSeries,
    ]);
    return featureView(context, args.seriesId);
  }

  computeFeaturesInto(context, args) {
    const outOffset = args.outOffset || 0;
    if (typeof this.exports.compute_features_into === 'function') {
      const directPtr = context.pointerForView(args.out, outOffset, FEATURE_COUNT);
      let staged = null;
      const outPtr = directPtr !== null ? directPtr : (staged = context.scratch.allocFloat64(FEATURE_COUNT)).ptr;
      this.runExport('compute_features_into', [
        context.arenaPtr,
        args.seriesId,
        args.windowSize,
        context.maxSeries,
        context.samplesPerSeries,
        outPtr,
      ]);
      if (staged) {
        args.out.set(staged.view, outOffset);
      }
      return args.out;
    }

    const internal = this.computeFeatures(context, args);
    args.out.set(internal, outOffset);
    return args.out;
  }

  computeFeaturesBatch(context, args) {
    const { ptr: idsPtr } = context.scratch.writeUint32(args.seriesIds, 0, args.count);
    const { ptr: windowsPtr } = context.scratch.writeUint32(args.windowSizes, 0, args.count);
    return this.runExport('compute_features_batch', [
      context.arenaPtr,
      idsPtr,
      windowsPtr,
      args.count,
      context.maxSeries,
      context.samplesPerSeries,
    ]);
  }

  computeFeaturesBatchInto(context, args) {
    const outOffset = args.outOffset || 0;
    const { ptr: idsPtr } = context.scratch.writeUint32(args.seriesIds, 0, args.count);
    const { ptr: windowsPtr } = context.scratch.writeUint32(args.windowSizes, 0, args.count);

    if (typeof this.exports.compute_features_batch_into === 'function') {
      const totalSlots = args.count * FEATURE_COUNT;
      const directPtr = context.pointerForView(args.out, outOffset, totalSlots);
      let staged = null;
      const outPtr = directPtr !== null ? directPtr : (staged = context.scratch.allocFloat64(totalSlots)).ptr;
      this.runExport('compute_features_batch_into', [
        context.arenaPtr,
        idsPtr,
        windowsPtr,
        args.count,
        context.maxSeries,
        context.samplesPerSeries,
        outPtr,
      ]);
      if (staged) {
        args.out.set(staged.view, outOffset);
      }
      return args.out;
    }

    this.computeFeaturesBatch(context, args);
    for (let i = 0; i < args.count; i += 1) {
      args.out.set(featureView(context, args.seriesIds[i]), outOffset + (i * FEATURE_COUNT));
    }
    return args.out;
  }

  computeFeaturesBatchFixed(context, args) {
    const { ptr: idsPtr } = context.scratch.writeUint32(args.seriesIds, 0, args.count);
    return this.runExport('compute_features_batch_fixed_window', [
      context.arenaPtr,
      idsPtr,
      args.count,
      args.windowSize,
      context.maxSeries,
      context.samplesPerSeries,
    ]);
  }

  computeFeaturesBatchFixedInto(context, args) {
    const outOffset = args.outOffset || 0;
    const { ptr: idsPtr } = context.scratch.writeUint32(args.seriesIds, 0, args.count);
    if (typeof this.exports.compute_features_batch_fixed_window_into === 'function') {
      const totalSlots = args.count * FEATURE_COUNT;
      const directPtr = context.pointerForView(args.out, outOffset, totalSlots);
      let staged = null;
      const outPtr = directPtr !== null ? directPtr : (staged = context.scratch.allocFloat64(totalSlots)).ptr;
      this.runExport('compute_features_batch_fixed_window_into', [
        context.arenaPtr,
        idsPtr,
        args.count,
        args.windowSize,
        context.maxSeries,
        context.samplesPerSeries,
        outPtr,
      ]);
      if (staged) {
        args.out.set(staged.view, outOffset);
      }
      return args.out;
    }

    this.computeFeaturesBatchFixed(context, args);
    for (let i = 0; i < args.count; i += 1) {
      args.out.set(featureView(context, args.seriesIds[i]), outOffset + (i * FEATURE_COUNT));
    }
    return args.out;
  }

  computePortfolioInto(context, args) {
    if (typeof this.exports.compute_portfolio_into === 'function') {
      const { ptr: valuesPtr } = context.scratch.writeFloat64(args.values, 0, args.values.length);
      const { ptr: baselinesPtr } = context.scratch.writeFloat64(args.baselines, 0, args.baselines.length);
      const aggregate = context.pointerOrScratch(args.aggregateOut, 0, RESULT_LAYOUTS.PORTFOLIO.count);
      const deviations = context.pointerOrScratch(args.deviationsOut, 0, args.values.length);
      const harvest = context.pointerOrScratch(args.harvestOut, 0, args.values.length);
      const rebalance = context.pointerOrScratch(args.rebalanceOut, 0, args.values.length);
      this.runExport('compute_portfolio_into', [
        valuesPtr,
        baselinesPtr,
        args.values.length,
        args.params.cashBalance,
        args.params.harvestTrigger,
        args.params.rebalanceTrigger,
        args.params.cpTriggerAssetPercent,
        args.params.cpTriggerMinNegativeDev,
        aggregate.ptr,
        deviations.ptr,
        harvest.ptr,
        rebalance.ptr,
      ]);
      aggregate.copyBack();
      deviations.copyBack();
      harvest.copyBack();
      rebalance.copyBack();
      return args.aggregateOut;
    }

    throw createKernelError(
      DREAMER_ERROR_CODES.BACKEND_UNAVAILABLE,
      'Explicit compute_portfolio_into export is required for the stable Dreamer boundary',
      { backend: this.kind },
    );
  }

  scanDefectsInto(context, args) {
    if (typeof this.exports.scan_defects_into === 'function') {
      const { ptr: pricesPtr } = context.scratch.writeFloat64(args.prices, 0, args.prices.length);
      const out = context.pointerOrScratch(args.out, 0, RESULT_LAYOUTS.DEFECT_SCAN.count);
      this.runExport('scan_defects_into', [
        pricesPtr,
        args.prices.length,
        args.rebalanceTrigger,
        args.crashThreshold,
        out.ptr,
      ]);
      out.copyBack();
      return args.out;
    }

    if (typeof this.exports.scan_defects === 'function' && typeof this.exports.get_f64_result === 'function') {
      const { ptr: pricesPtr } = context.scratch.writeFloat64(args.prices, 0, args.prices.length);
      this.runExport('scan_defects', [pricesPtr, args.prices.length, args.rebalanceTrigger, args.crashThreshold]);
      args.out[0] = this.exports.get_f64_result(0);
      args.out[1] = this.exports.get_f64_result(1);
      args.out[2] = this.exports.get_f64_result(2);
      return args.out;
    }

    throw createKernelError(
      DREAMER_ERROR_CODES.BACKEND_UNAVAILABLE,
      'scan_defects_into export is unavailable',
      { backend: this.kind },
    );
  }

  computeRegimeInto(context, args) {
    if (typeof this.exports.compute_regime_into === 'function') {
      const { ptr: historyPtr } = context.scratch.writeFloat64(args.history, 0, args.history.length);
      const out = context.pointerOrScratch(args.out, 0, RESULT_LAYOUTS.REGIME.count);
      this.runExport('compute_regime_into', [
        historyPtr,
        args.history.length,
        args.currentPrice,
        args.startPrice,
        out.ptr,
      ]);
      out.copyBack();
      return args.out;
    }

    if (typeof this.exports.compute_regime_export === 'function' && typeof this.exports.get_f64_result === 'function') {
      const { ptr: historyPtr } = context.scratch.writeFloat64(args.history, 0, args.history.length);
      this.runExport('compute_regime_export', [historyPtr, args.history.length, args.currentPrice, args.startPrice]);
      args.out[0] = this.exports.get_f64_result(0);
      args.out[1] = this.exports.get_f64_result(1);
      args.out[2] = this.exports.get_f64_result(2);
      args.out[3] = this.exports.get_f64_result(3);
      return args.out;
    }

    throw createKernelError(
      DREAMER_ERROR_CODES.BACKEND_UNAVAILABLE,
      'compute_regime_into export is unavailable',
      { backend: this.kind },
    );
  }
}

function createJsNumericCoreBackend() {
  return new JsFloat64NumericCoreBackend();
}

function createWasmNumericCoreBackend(exports, mode) {
  return new WasmNumericCoreBackend(exports, mode);
}

module.exports = Object.freeze({
  DreamerKernelError,
  createKernelError,
  ensureReadableFloat64,
  ensureReadableUint32,
  ensureWritableFloat64,
  createJsNumericCoreBackend,
  createWasmNumericCoreBackend,
  computeFeatureVectorInto,
  computePortfolioInto,
  scanDefectsInto,
  computeRegimeInto,
  JsFloat64NumericCoreBackend,
  WasmNumericCoreBackend,
  featureView,
  sameTypedRange,
});
