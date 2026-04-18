'use strict';

const ABI_VERSION = 100;
const CONTRACT_VERSION = '1.0.0';
const ALIGNMENT_BYTES = 16;
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

const FEATURE_INDICES = Object.freeze({
  MEAN_PRICE: 0,
  PRICE_VARIANCE: 1,
  VWAP: 2,
  LATEST_PRICE: 3,
  PRICE_MOMENTUM: 4,
  MEAN_VOLUME: 5,
});

const REGIME_CODES = Object.freeze({
  UNKNOWN: 0,
  CRAB_CHOP: 1,
  BULL_RUSH: 2,
  BEAR_CRASH: 3,
  STEADY_GROWTH: 4,
  VOLATILE_CHOP: 5,
});

const DREAMER_ERROR_CODES = Object.freeze({
  INVALID_ARGUMENT: 'ERR_INVALID_ARGUMENT',
  RANGE: 'ERR_RANGE',
  TYPE: 'ERR_TYPE',
  BACKEND_UNAVAILABLE: 'ERR_BACKEND_UNAVAILABLE',
  ABI_MISMATCH: 'ERR_ABI_MISMATCH',
  WASM_TRAP: 'ERR_WASM_TRAP',
  SCRATCH_CAPACITY: 'ERR_SCRATCH_CAPACITY',
  FALLBACK_EXHAUSTED: 'ERR_FALLBACK_EXHAUSTED',
});

const RUNTIME_MODES = Object.freeze({
  WASM_SIMD: 'wasm-simd',
  WASM_SCALAR: 'wasm-scalar',
  JS_FLOAT64_CORE: 'js-float64-core',
});

const RESULT_LAYOUTS = Object.freeze({
  PORTFOLIO: Object.freeze({
    count: 5,
    fieldNames: Object.freeze([
      'deviation_percent',
      'crash_active',
      'declining_count',
      'managed_baseline',
      'baseline_diff',
    ]),
    indices: Object.freeze({
      DEVIATION_PERCENT: 0,
      CRASH_ACTIVE: 1,
      DECLINING_COUNT: 2,
      MANAGED_BASELINE: 3,
      BASELINE_DIFF: 4,
    }),
  }),
  DEFECT_SCAN: Object.freeze({
    count: 3,
    fieldNames: Object.freeze([
      'is_defective',
      'max_drawdown',
      'trigger_hits',
    ]),
    indices: Object.freeze({
      IS_DEFECTIVE: 0,
      MAX_DRAWDOWN: 1,
      TRIGGER_HITS: 2,
    }),
  }),
  REGIME: Object.freeze({
    count: 4,
    fieldNames: Object.freeze([
      'regime_code',
      'roi',
      'volatility',
      'mean',
    ]),
    indices: Object.freeze({
      REGIME_CODE: 0,
      ROI: 1,
      VOLATILITY: 2,
      MEAN: 3,
    }),
  }),
});

function computeOffsets(maxSeries, samplesPerSeries) {
  const metaStart = CONTROL_SLOTS;
  const metaEnd = metaStart + (maxSeries * META_SLOTS_PER_SERIES);
  const pricesStart = metaEnd;
  const pricesEnd = pricesStart + (maxSeries * samplesPerSeries);
  const volumesStart = pricesEnd;
  const volumesEnd = volumesStart + (maxSeries * samplesPerSeries);
  const featuresStart = volumesEnd;
  const featuresEnd = featuresStart + (maxSeries * FEATURE_COUNT);

  return Object.freeze({
    control: 0,
    metaStart,
    metaEnd,
    pricesStart,
    pricesEnd,
    volumesStart,
    volumesEnd,
    featuresStart,
    featuresEnd,
  });
}

function computeTotalSlots(maxSeries, samplesPerSeries) {
  return computeOffsets(maxSeries, samplesPerSeries).featuresEnd;
}

function computeBytes(maxSeries, samplesPerSeries) {
  return computeTotalSlots(maxSeries, samplesPerSeries) * FLOAT64_BYTES;
}

const DREAMER_BUFFER_LAYOUT = Object.freeze({
  CONTROL: Object.freeze({
    maxSeries: 0,
    samplesPerSeries: 1,
    featureCount: 2,
    lastSeriesId: 3,
    lastPrice: 4,
    lastVolume: 5,
    totalSlots: CONTROL_SLOTS,
  }),
  SERIES_META: Object.freeze({
    slotsPerSeries: META_SLOTS_PER_SERIES,
    writeIndex: 0,
    sampleCount: 1,
  }),
  FEATURE_COUNT,
  FEATURE_INDICES,
  RESULT_LAYOUTS,
  ALIGNMENT: Object.freeze({
    bytes: ALIGNMENT_BYTES,
    float64Slots: ALIGNMENT_BYTES / FLOAT64_BYTES,
  }),
  computeOffsets,
  computeTotalSlots,
  computeBytes,
});

const FEATURE_TRANSFORMS = Object.freeze([
  'mean price over the requested rolling window',
  'population variance over the requested rolling window',
  'VWAP from the same price and volume window',
  'latest price copy-out from the window tail',
  'price momentum as latest minus oldest price in the window',
  'mean volume over the requested rolling window',
]);

const INDICATOR_TRANSFORMS = Object.freeze([
  'ring-buffer state update for per-series price and volume streams',
  'batched rolling-window feature extraction over many series/window pairs',
  'portfolio deviation, crash-protection, harvest, and rebalance scans',
  'defect scan over historical prices for trigger-hit and drawdown detection',
  'regime statistics: ROI, mean, volatility, and regime classification',
]);

const JS_RESPONSIBILITIES = Object.freeze([
  'network ingress, scheduling, retries, persistence, and logging',
  'symbol-to-series mapping and other object-heavy orchestration concerns',
  'typed-array staging for values extracted from Dreamer object graphs',
  'backend selection, capability detection, and fallback activation',
  'reading result buffers after the numeric core has written them',
]);

const JS_MUST_NOT_TOUCH = Object.freeze([
  'rolling sums, rolling sums of squares, VWAP, variance, or momentum math',
  'portfolio deviation percentages and crash-protection threshold math',
  'defect scan trigger-hit or drawdown math',
  'regime ROI, mean, variance, or volatility math',
  'hot-path ring-buffer index arithmetic once numeric scalars and typed buffers are staged',
]);

const NUMERIC_CORE_RESPONSIBILITIES = Object.freeze([
  'all precision-sensitive arithmetic on f64 values',
  'all ring-buffer mutation on the hot path',
  'all batch reductions and fixed-stride output writes',
  'all threshold comparisons that influence numeric outputs',
  'all result-buffer population for features, portfolio, regime, and defect outputs',
]);

const HOT_PATH_SCOPE = Object.freeze([
  Object.freeze({
    id: 'ring-buffer-state-update',
    signedOff: true,
    owner: 'numeric-core',
    batchingUnit: '1 tick or 1 contiguous tick batch',
    transforms: Object.freeze([
      'write-index wrap arithmetic',
      'sample-count saturation',
      'contiguous price slab writes',
      'contiguous volume slab writes',
    ]),
    stateInputs: Object.freeze([
      'arena.control',
      'arena.seriesMeta[seriesId]',
      'arena.prices[seriesId][samplesPerSeries]',
      'arena.volumes[seriesId][samplesPerSeries]',
    ]),
    outputs: Object.freeze([
      'updated series metadata',
      'updated price and volume slabs',
      'updated control header lastSeriesId/lastPrice/lastVolume',
    ]),
    jsStopPoint: 'After JS resolves seriesId, price, volume, or typed batch views, it must dispatch ingest_tick / ingest_batch and stop touching ring math.',
    excludedFromJs: Object.freeze([
      'portfolioSummary field traversal',
      'symbol lookup',
      'JSON serialization',
      'logging and telemetry',
    ]),
  }),
  Object.freeze({
    id: 'rolling-window-feature-reduction',
    signedOff: true,
    owner: 'numeric-core',
    batchingUnit: '1 series-window pair',
    transforms: FEATURE_TRANSFORMS,
    stateInputs: Object.freeze([
      'arena.seriesMeta[seriesId]',
      'arena.prices[seriesId][samplesPerSeries]',
      'arena.volumes[seriesId][samplesPerSeries]',
      'windowSize',
    ]),
    outputs: Object.freeze([
      'Float64 feature vector with fixed FEATURE_COUNT stride',
    ]),
    jsStopPoint: 'After JS selects seriesId, windowSize, and the output buffer, the numeric core owns the full reduction.',
    excludedFromJs: Object.freeze([
      'Array.prototype.reduce over object histories',
      'variance / VWAP / momentum math in orchestration code',
    ]),
  }),
  Object.freeze({
    id: 'batched-feature-extraction',
    signedOff: true,
    owner: 'numeric-core',
    batchingUnit: 'N rows of {seriesId, windowSize}',
    transforms: Object.freeze([
      'fixed-stride FEATURE_COUNT writes',
      'row-wise feature reduction over typed series/window vectors',
    ]),
    stateInputs: Object.freeze([
      'Uint32Array seriesIds',
      'Uint32Array windowSizes',
      'arena series slabs',
    ]),
    outputs: Object.freeze([
      'Float64Array out with N * FEATURE_COUNT slots',
    ]),
    jsStopPoint: 'Once typed batch vectors are staged, JS must only dispatch and then read the output slab.',
    excludedFromJs: Object.freeze([
      'per-row object allocation in the hot loop',
      'feature decoding inside the reduction loop',
    ]),
  }),
  Object.freeze({
    id: 'portfolio-vector-scan',
    signedOff: true,
    owner: 'numeric-core',
    batchingUnit: 'N assets in one portfolio snapshot',
    transforms: Object.freeze([
      'per-asset deviation computation',
      'harvest-candidate marking',
      'rebalance-candidate marking',
      'aggregate crash-protection metrics',
    ]),
    stateInputs: Object.freeze([
      'Float64Array values',
      'Float64Array baselines',
      'cashBalance',
      'harvestTrigger',
      'rebalanceTrigger',
      'cpTriggerAssetPercent',
      'cpTriggerMinNegativeDev',
    ]),
    outputs: Object.freeze([
      'portfolio aggregate Float64Array[5]',
      'deviations Float64Array[N]',
      'harvest flags Float64Array[N]',
      'rebalance flags Float64Array[N]',
    ]),
    jsStopPoint: 'After JS extracts Value and Baseline fields into typed arrays, the numeric core owns all deviation and threshold math.',
    excludedFromJs: Object.freeze([
      'portfolioSummary.forEach deviation math',
      'crash-protection threshold comparisons in orchestration code',
    ]),
  }),
  Object.freeze({
    id: 'defect-scan',
    signedOff: true,
    owner: 'numeric-core',
    batchingUnit: '1 price history vector',
    transforms: Object.freeze([
      'trigger-hit counting',
      'look-ahead drawdown search',
      'defect flag derivation',
    ]),
    stateInputs: Object.freeze([
      'Float64Array prices',
      'rebalanceTrigger',
      'crashThreshold',
    ]),
    outputs: Object.freeze([
      'defect result Float64Array[3]',
    ]),
    jsStopPoint: 'After JS stages the price history buffer, the numeric core owns all trigger and drawdown math.',
    excludedFromJs: Object.freeze([
      'nested object-history scans on the hot path',
      'drawdown math in orchestration code',
    ]),
  }),
  Object.freeze({
    id: 'regime-statistics',
    signedOff: true,
    owner: 'numeric-core',
    batchingUnit: '1 history vector',
    transforms: Object.freeze([
      'ROI computation',
      'mean and variance reduction',
      'volatility derivation',
      'regime classification thresholds',
    ]),
    stateInputs: Object.freeze([
      'Float64Array history',
      'currentPrice',
      'startPrice',
    ]),
    outputs: Object.freeze([
      'regime result Float64Array[4]',
    ]),
    jsStopPoint: 'After JS stages the history vector and scalar anchors, the numeric core owns all regime math.',
    excludedFromJs: Object.freeze([
      'Array.prototype.reduce over history objects',
      'volatility / regime threshold math outside the numeric core',
    ]),
  }),
]);

const COMMON_ERROR_SEMANTICS = Object.freeze({
  invalidArguments: 'Throw DreamerKernelError with ERR_INVALID_ARGUMENT / ERR_RANGE / ERR_TYPE before invoking any backend.',
  backendUnavailable: 'If the preferred backend is missing required exports or reports an ABI mismatch, activate the next fallback mode.',
  wasmTrap: 'If a wasm backend traps, surface ERR_WASM_TRAP and replay once on the JS Float64 core when fallback is enabled.',
  fallbackExhausted: 'If no backend can satisfy the call, throw ERR_FALLBACK_EXHAUSTED without mutating caller-owned output buffers.',
});

const BUFFER_OWNERSHIP = Object.freeze({
  arena: 'Adapter-owned, long-lived ring-state buffer. The numeric core mutates only the documented arena sections in place.',
  callerInput: 'Caller-owned and read-only for the duration of the call. The backend may copy into scratch memory but must not retain references.',
  callerOutput: 'Caller-owned and write-only from the numeric core perspective. The adapter may read after the call completes.',
  scratch: 'Adapter-owned ephemeral staging buffer. Used only for backend transport, never as external API state.',
});

const CALL_BOUNDARY = Object.freeze({
  ingest_tick: Object.freeze({
    exportName: 'ingest_tick',
    inputs: Object.freeze([
      'u32 seriesId',
      'f64 price',
      'f64 volume',
    ]),
    batchingShape: 'single tick',
    stateInputs: HOT_PATH_SCOPE[0].stateInputs,
    outputs: HOT_PATH_SCOPE[0].outputs,
    bufferOwnership: BUFFER_OWNERSHIP,
    errorSemantics: COMMON_ERROR_SEMANTICS,
    fallback: 'wasm-simd -> wasm-scalar -> js-float64-core',
  }),
  ingest_batch: Object.freeze({
    exportName: 'ingest_batch',
    inputs: Object.freeze([
      'u32 seriesId',
      'Float64Array prices[count]',
      'Float64Array volumes[count]',
      'u32 priceOffset',
      'u32 volumeOffset',
      'u32 count',
    ]),
    batchingShape: 'one contiguous batch for one series, split into tail/head spans only if the ring wraps once',
    stateInputs: HOT_PATH_SCOPE[0].stateInputs,
    outputs: HOT_PATH_SCOPE[0].outputs,
    bufferOwnership: BUFFER_OWNERSHIP,
    errorSemantics: COMMON_ERROR_SEMANTICS,
    fallback: 'wasm-simd -> wasm-scalar -> js-float64-core',
  }),
  compute_features_into: Object.freeze({
    exportName: 'compute_features_into',
    inputs: Object.freeze([
      'u32 seriesId',
      'u32 windowSize',
      'Float64Array out[FEATURE_COUNT]',
      'u32 outOffset',
    ]),
    batchingShape: 'single series-window pair',
    stateInputs: HOT_PATH_SCOPE[1].stateInputs,
    outputs: HOT_PATH_SCOPE[1].outputs,
    bufferOwnership: BUFFER_OWNERSHIP,
    errorSemantics: COMMON_ERROR_SEMANTICS,
    fallback: 'wasm-simd -> wasm-scalar -> js-float64-core',
  }),
  compute_features_batch_into: Object.freeze({
    exportName: 'compute_features_batch_into',
    inputs: Object.freeze([
      'Uint32Array seriesIds[count]',
      'Uint32Array windowSizes[count]',
      'Float64Array out[count * FEATURE_COUNT]',
      'u32 outOffset',
      'u32 count',
    ]),
    batchingShape: 'N rows with fixed FEATURE_COUNT output stride',
    stateInputs: HOT_PATH_SCOPE[2].stateInputs,
    outputs: HOT_PATH_SCOPE[2].outputs,
    bufferOwnership: BUFFER_OWNERSHIP,
    errorSemantics: COMMON_ERROR_SEMANTICS,
    fallback: 'wasm-simd -> wasm-scalar -> js-float64-core',
  }),
  compute_features_batch_fixed_window_into: Object.freeze({
    exportName: 'compute_features_batch_fixed_window_into',
    inputs: Object.freeze([
      'Uint32Array seriesIds[count]',
      'u32 windowSize',
      'Float64Array out[count * FEATURE_COUNT]',
      'u32 outOffset',
      'u32 count',
    ]),
    batchingShape: 'N rows sharing one fixed window size',
    stateInputs: HOT_PATH_SCOPE[2].stateInputs,
    outputs: HOT_PATH_SCOPE[2].outputs,
    bufferOwnership: BUFFER_OWNERSHIP,
    errorSemantics: COMMON_ERROR_SEMANTICS,
    fallback: 'wasm-simd -> wasm-scalar -> js-float64-core',
  }),
  compute_portfolio_into: Object.freeze({
    exportName: 'compute_portfolio_into',
    inputs: Object.freeze([
      'Float64Array values[assetCount]',
      'Float64Array baselines[assetCount]',
      'u32 assetCount',
      'f64 cashBalance',
      'f64 harvestTrigger',
      'f64 rebalanceTrigger',
      'f64 cpTriggerAssetPercent',
      'f64 cpTriggerMinNegativeDev',
      'Float64Array aggregateOut[5]',
      'Float64Array deviationsOut[assetCount]',
      'Float64Array harvestOut[assetCount]',
      'Float64Array rebalanceOut[assetCount]',
    ]),
    batchingShape: 'one asset vector per portfolio snapshot',
    stateInputs: HOT_PATH_SCOPE[3].stateInputs,
    outputs: HOT_PATH_SCOPE[3].outputs,
    bufferOwnership: BUFFER_OWNERSHIP,
    errorSemantics: COMMON_ERROR_SEMANTICS,
    fallback: 'wasm-simd -> wasm-scalar -> js-float64-core',
  }),
  scan_defects_into: Object.freeze({
    exportName: 'scan_defects_into',
    inputs: Object.freeze([
      'Float64Array prices[priceCount]',
      'u32 priceCount',
      'f64 rebalanceTrigger',
      'f64 crashThreshold',
      'Float64Array out[3]',
    ]),
    batchingShape: 'one price history vector',
    stateInputs: HOT_PATH_SCOPE[4].stateInputs,
    outputs: HOT_PATH_SCOPE[4].outputs,
    bufferOwnership: BUFFER_OWNERSHIP,
    errorSemantics: COMMON_ERROR_SEMANTICS,
    fallback: 'wasm-simd -> wasm-scalar -> js-float64-core',
  }),
  compute_regime_into: Object.freeze({
    exportName: 'compute_regime_into',
    inputs: Object.freeze([
      'Float64Array history[historyLen]',
      'u32 historyLen',
      'f64 currentPrice',
      'f64 startPrice',
      'Float64Array out[4]',
    ]),
    batchingShape: 'one history vector',
    stateInputs: HOT_PATH_SCOPE[5].stateInputs,
    outputs: HOT_PATH_SCOPE[5].outputs,
    bufferOwnership: BUFFER_OWNERSHIP,
    errorSemantics: COMMON_ERROR_SEMANTICS,
    fallback: 'wasm-simd -> wasm-scalar -> js-float64-core',
  }),
});

function normalizeCapabilities(capabilities = {}) {
  const runtimeFamily = typeof capabilities.runtimeFamily === 'string'
    ? capabilities.runtimeFamily
    : (typeof process === 'object' && process !== null && process.versions && process.versions.node
      ? 'node'
      : (typeof window === 'object' ? 'browser' : 'unknown'));

  return Object.freeze({
    hasWasm: typeof capabilities.hasWasm === 'boolean'
      ? capabilities.hasWasm
      : Boolean(typeof WebAssembly === 'object' && WebAssembly !== null),
    simd128: Boolean(capabilities.simd128),
    float64: typeof capabilities.float64 === 'boolean'
      ? capabilities.float64
      : Boolean(typeof Float64Array === 'function'),
    runtimeFamily,
  });
}

function createRuntimePlan(capabilities = {}) {
  const caps = normalizeCapabilities(capabilities);
  if (!caps.float64) {
    return Object.freeze({
      mode: 'unsupported',
      reason: 'Float64Array support is required for the Dreamer numeric boundary',
      fallback: null,
      capabilities: caps,
    });
  }
  if (!caps.hasWasm) {
    return Object.freeze({
      mode: RUNTIME_MODES.JS_FLOAT64_CORE,
      reason: 'No WebAssembly runtime is available',
      fallback: null,
      capabilities: caps,
    });
  }
  if (caps.simd128) {
    return Object.freeze({
      mode: RUNTIME_MODES.WASM_SIMD,
      reason: 'SIMD-capable WebAssembly runtime detected',
      fallback: RUNTIME_MODES.WASM_SCALAR,
      capabilities: caps,
    });
  }
  return Object.freeze({
    mode: RUNTIME_MODES.WASM_SCALAR,
    reason: 'WebAssembly is available but SIMD is not',
    fallback: RUNTIME_MODES.JS_FLOAT64_CORE,
    capabilities: caps,
  });
}

const FALLBACK_POLICY = Object.freeze({
  order: Object.freeze([
    RUNTIME_MODES.WASM_SIMD,
    RUNTIME_MODES.WASM_SCALAR,
    RUNTIME_MODES.JS_FLOAT64_CORE,
  ]),
  activationRules: Object.freeze([
    'Prefer wasm-simd when simd128 is available and the ABI matches.',
    'Drop to wasm-scalar when wasm exists but simd128 is unavailable.',
    'Drop to js-float64-core when no wasm backend is attached or when wasm traps and fallback is enabled.',
    'Never fall back on invalid caller input; validation errors are surfaced immediately.',
  ]),
});

const DREAMER_WASM_BOUNDARY = Object.freeze({
  version: CONTRACT_VERSION,
  abiVersion: ABI_VERSION,
  precision: 'binary64',
  alignmentBytes: ALIGNMENT_BYTES,
  signedOff: HOT_PATH_SCOPE.every((entry) => entry.signedOff),
  jsResponsibilities: JS_RESPONSIBILITIES,
  jsMustNotTouch: JS_MUST_NOT_TOUCH,
  numericCoreResponsibilities: NUMERIC_CORE_RESPONSIBILITIES,
  featureTransforms: FEATURE_TRANSFORMS,
  indicatorTransforms: INDICATOR_TRANSFORMS,
  hotPathScope: HOT_PATH_SCOPE,
  callBoundary: CALL_BOUNDARY,
  fallbackPolicy: FALLBACK_POLICY,
  runtimeModes: RUNTIME_MODES,
  promotedKernels: CALL_BOUNDARY,
});

module.exports = Object.freeze({
  ABI_VERSION,
  CONTRACT_VERSION,
  ALIGNMENT_BYTES,
  FLOAT64_BYTES,
  CONTROL_SLOTS,
  META_SLOTS_PER_SERIES,
  FEATURE_COUNT,
  FEATURE_NAMES,
  FEATURE_INDICES,
  REGIME_CODES,
  RESULT_LAYOUTS,
  DREAMER_ERROR_CODES,
  RUNTIME_MODES,
  DREAMER_BUFFER_LAYOUT,
  DREAMER_WASM_BOUNDARY,
  DREAMER_KERNEL_CONTRACT: DREAMER_WASM_BOUNDARY,
  HOT_PATH_SCOPE,
  CALL_BOUNDARY,
  JS_RESPONSIBILITIES,
  JS_MUST_NOT_TOUCH,
  NUMERIC_CORE_RESPONSIBILITIES,
  BUFFER_OWNERSHIP,
  FALLBACK_POLICY,
  normalizeCapabilities,
  createRuntimePlan,
});
