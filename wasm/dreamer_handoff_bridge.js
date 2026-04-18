'use strict';

const {
  FEATURE_COUNT,
  RESULT_LAYOUTS,
} = require('./dreamer_kernel_contract.js');
const { DreamerKernelAdapter } = require('./dreamer_kernel_adapter.js');

function asArray(value, label) {
  if (!Array.isArray(value)) {
    throw new TypeError(`${label} must be an array`);
  }
  return value;
}

function readFiniteNumber(source, key, label, fallback = undefined) {
  const value = source[key];
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (fallback !== undefined) {
    return fallback;
  }
  throw new TypeError(`${label} must be a finite number`);
}

class DreamerHandoffBridge {
  constructor(adapter, options = {}) {
    if (!(adapter instanceof DreamerKernelAdapter)) {
      throw new TypeError('DreamerHandoffBridge requires a DreamerKernelAdapter instance');
    }
    this.adapter = adapter;
    this.symbolField = options.symbolField || 'Symbol';
    this.priceField = options.priceField || 'Price';
    this.volumeField = options.volumeField || 'Volume';
    this.valueField = options.valueField || 'Value';
    this.baselineField = options.baselineField || 'Baseline';
    this.defaultVolume = typeof options.defaultVolume === 'number' ? options.defaultVolume : 1;
    this.seriesBySymbol = new Map();
    this.symbolBySeries = [];
    this._seriesIds = new Uint32Array(0);
    this._windowSizes = new Uint32Array(0);
    this._featureOut = new Float64Array(0);
    this._valueStaging = new Float64Array(0);
    this._baselineStaging = new Float64Array(0);
    this._historyStaging = new Float64Array(0);
    this._portfolioAggregate = new Float64Array(RESULT_LAYOUTS.PORTFOLIO.count);
    this._portfolioDeviations = new Float64Array(0);
    this._portfolioHarvest = new Float64Array(0);
    this._portfolioRebalance = new Float64Array(0);
    this._defectOut = new Float64Array(RESULT_LAYOUTS.DEFECT_SCAN.count);
    this._regimeOut = new Float64Array(RESULT_LAYOUTS.REGIME.count);
  }

  ensureSeriesId(symbol) {
    if (typeof symbol !== 'string' || symbol.length === 0) {
      throw new TypeError('symbol must be a non-empty string');
    }
    if (this.seriesBySymbol.has(symbol)) {
      return this.seriesBySymbol.get(symbol);
    }
    const nextSeriesId = this.symbolBySeries.length;
    if (nextSeriesId >= this.adapter.maxSeries) {
      throw new RangeError(`cannot register ${symbol}: maxSeries=${this.adapter.maxSeries} exhausted`);
    }
    this.seriesBySymbol.set(symbol, nextSeriesId);
    this.symbolBySeries.push(symbol);
    return nextSeriesId;
  }

  getSeriesId(symbol) {
    return this.ensureSeriesId(symbol);
  }

  getSymbol(seriesId) {
    return this.symbolBySeries[seriesId] || null;
  }

  _ensureUint32Buffer(current, length) {
    return current.length >= length ? current : new Uint32Array(length);
  }

  _ensureFloat64Buffer(current, length) {
    return current.length >= length ? current : new Float64Array(length);
  }

  ingestPortfolioSummary(portfolioSummary, options = {}) {
    const rows = asArray(portfolioSummary, 'portfolioSummary');
    const symbolField = options.symbolField || this.symbolField;
    const priceField = options.priceField || this.priceField;
    const volumeField = options.volumeField || this.volumeField;
    const defaultVolume = typeof options.defaultVolume === 'number' ? options.defaultVolume : this.defaultVolume;

    let ingested = 0;
    for (const row of rows) {
      const symbol = row[symbolField];
      const price = readFiniteNumber(row, priceField, `${symbol}.${priceField}`);
      const volume = readFiniteNumber(row, volumeField, `${symbol}.${volumeField}`, defaultVolume);
      const seriesId = this.ensureSeriesId(symbol);
      this.adapter.ingestTick(seriesId, price, volume);
      ingested += 1;
    }
    return ingested;
  }

  computeFeatureBatch(requests) {
    const rows = asArray(requests, 'requests');
    this._seriesIds = this._ensureUint32Buffer(this._seriesIds, rows.length);
    this._windowSizes = this._ensureUint32Buffer(this._windowSizes, rows.length);
    this._featureOut = this._ensureFloat64Buffer(this._featureOut, rows.length * FEATURE_COUNT);

    for (let i = 0; i < rows.length; i += 1) {
      const request = rows[i];
      const seriesId = request.seriesId !== undefined
        ? request.seriesId
        : this.ensureSeriesId(request.symbol);
      this._seriesIds[i] = seriesId;
      this._windowSizes[i] = request.windowSize;
    }

    const seriesIds = this._seriesIds.subarray(0, rows.length);
    const windowSizes = this._windowSizes.subarray(0, rows.length);
    const out = this._featureOut.subarray(0, rows.length * FEATURE_COUNT);
    this.adapter.computeFeaturesBatchInto(seriesIds, windowSizes, out, 0, rows.length);
    return out;
  }

  computePortfolioFromSummary(portfolioSummary, params = {}) {
    const rows = asArray(portfolioSummary, 'portfolioSummary');
    const valueField = params.valueField || this.valueField;
    const baselineField = params.baselineField || this.baselineField;
    const exclusions = new Set(params.exclusions || []);
    const includedSymbols = [];

    this._valueStaging = this._ensureFloat64Buffer(this._valueStaging, rows.length);
    this._baselineStaging = this._ensureFloat64Buffer(this._baselineStaging, rows.length);
    this._portfolioDeviations = this._ensureFloat64Buffer(this._portfolioDeviations, rows.length);
    this._portfolioHarvest = this._ensureFloat64Buffer(this._portfolioHarvest, rows.length);
    this._portfolioRebalance = this._ensureFloat64Buffer(this._portfolioRebalance, rows.length);

    let count = 0;
    for (const row of rows) {
      const symbol = row[this.symbolField];
      if (exclusions.has(symbol)) {
        continue;
      }
      this.ensureSeriesId(symbol);
      this._valueStaging[count] = readFiniteNumber(row, valueField, `${symbol}.${valueField}`);
      this._baselineStaging[count] = readFiniteNumber(row, baselineField, `${symbol}.${baselineField}`, 0);
      includedSymbols.push(symbol);
      count += 1;
    }

    const values = this._valueStaging.subarray(0, count);
    const baselines = this._baselineStaging.subarray(0, count);
    const deviations = this._portfolioDeviations.subarray(0, count);
    const harvest = this._portfolioHarvest.subarray(0, count);
    const rebalance = this._portfolioRebalance.subarray(0, count);

    this.adapter.computePortfolioInto(values, baselines, params, this._portfolioAggregate, deviations, harvest, rebalance);
    return {
      symbols: includedSymbols,
      aggregate: this._portfolioAggregate,
      deviations,
      harvestCandidates: harvest,
      rebalanceCandidates: rebalance,
    };
  }

  _stageHistory(historyLike, priceField = this.priceField) {
    if (historyLike instanceof Float64Array) {
      return historyLike;
    }
    if (Array.isArray(historyLike)) {
      this._historyStaging = this._ensureFloat64Buffer(this._historyStaging, historyLike.length);
      for (let i = 0; i < historyLike.length; i += 1) {
        const entry = historyLike[i];
        if (typeof entry === 'number') {
          this._historyStaging[i] = entry;
        } else {
          this._historyStaging[i] = readFiniteNumber(entry, priceField, `history[${i}].${priceField}`);
        }
      }
      return this._historyStaging.subarray(0, historyLike.length);
    }
    throw new TypeError('historyLike must be an array or Float64Array');
  }

  computeRegimeFromHistory(historyLike, options = {}) {
    const priceField = options.priceField || this.priceField;
    const history = this._stageHistory(historyLike, priceField);
    const currentPrice = options.currentPrice !== undefined ? options.currentPrice : history[history.length - 1];
    const startPrice = options.startPrice !== undefined ? options.startPrice : history[0];
    this.adapter.computeRegimeInto(history, currentPrice, startPrice, this._regimeOut);
    return this._regimeOut;
  }

  scanDefectsFromHistory(historyLike, options = {}) {
    const priceField = options.priceField || this.priceField;
    const history = this._stageHistory(historyLike, priceField);
    this.adapter.scanDefectsInto(
      history,
      options.rebalanceTrigger,
      options.crashThreshold === undefined ? 0.01 : options.crashThreshold,
      this._defectOut,
    );
    return this._defectOut;
  }
}

module.exports = {
  DreamerHandoffBridge,
};
