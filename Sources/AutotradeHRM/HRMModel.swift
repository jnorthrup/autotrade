//
//  HRMModel.swift
//  AutotradeHRM
//
//  Hierarchical edge predictor with explicit H/L carry, rotary growth, and
//  checkpointed growth metadata.
//

import Foundation

public let allowedGrowthPowers = [4, 16, 64, 256]

public protocol HRMEdgeBackend {
    mutating func registerEdges(_ edges: [String])
    mutating func predict(edge: String, fisheye: [Double], carry: HRMCarryState?) -> HRMPredictionStep
    mutating func train(edge: String, fisheye: [Double], targets: HRMTargets, carry: HRMCarryState?) -> HRMTrainingStep
    mutating func grow(dim: GrowthDimension, geometryPolicy: GrowthGeometryPolicy?)
    func save(path: String) throws
    mutating func load(path: String) throws
}

public enum HRMCheckpointError: Error, Equatable, LocalizedError {
    case fileMissing(path: String)
    case encodeFailed(path: String, reason: String)
    case writeFailed(path: String, reason: String)
    case readFailed(path: String, reason: String)
    case decodeFailed(path: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case let .fileMissing(path):
            return "HRM checkpoint missing at \(path)"
        case let .encodeFailed(path, reason):
            return "HRM checkpoint encode failed for \(path): \(reason)"
        case let .writeFailed(path, reason):
            return "HRM checkpoint write failed for \(path): \(reason)"
        case let .readFailed(path, reason):
            return "HRM checkpoint read failed for \(path): \(reason)"
        case let .decodeFailed(path, reason):
            return "HRM checkpoint decode failed for \(path): \(reason)"
        }
    }
}

public enum GrowthDimension: String, Codable, CaseIterable, Sendable {
    case h
    case z
    case H
    case L
}

public enum GrowthGeometryPolicy: String, Codable, CaseIterable, Sendable {
    case quadrantRotation
    case axialFallback0_180_180_0
    case randomRestart
    case bridgeCompatibility
}

public struct HRMCarryState: Codable, Equatable, Sendable {
    public var zH: [Double]
    public var zL: [Double]

    public init(zH: [Double], zL: [Double]) {
        self.zH = zH
        self.zL = zL
    }

    public func fitted(hDim: Int, zDim: Int, fallbackH: [Double], fallbackL: [Double]) -> HRMCarryState {
        HRMCarryState(
            zH: fitVector(zH, count: hDim, fallback: fallbackH),
            zL: fitVector(zL, count: zDim, fallback: fallbackL)
        )
    }
}

public struct HRMEdgeOutputs: Codable, Equatable, Sendable {
    public let fraction: Double
    public let ptt: Double
    public let stop: Double

    public init(fraction: Double, ptt: Double, stop: Double) {
        self.fraction = fraction
        self.ptt = ptt
        self.stop = stop
    }
}

public struct HRMTargets: Codable, Equatable, Sendable {
    public let fraction: Double
    public let ptt: Double
    public let stop: Double

    public init(fraction: Double, ptt: Double, stop: Double) {
        self.fraction = fraction
        self.ptt = ptt
        self.stop = stop
    }
}

public struct HRMExecutionTrace: Codable, Equatable, Sendable {
    public let hPasses: Int
    public let lPasses: Int
    public let cycleOrder: [String]

    public init(hPasses: Int, lPasses: Int, cycleOrder: [String]) {
        self.hPasses = hPasses
        self.lPasses = lPasses
        self.cycleOrder = cycleOrder
    }
}

public struct HRMPredictionStep: Codable, Equatable, Sendable {
    public let outputs: HRMEdgeOutputs
    public let nextCarry: HRMCarryState
    public let trace: HRMExecutionTrace

    public init(outputs: HRMEdgeOutputs, nextCarry: HRMCarryState, trace: HRMExecutionTrace) {
        self.outputs = outputs
        self.nextCarry = nextCarry
        self.trace = trace
    }
}

public struct HRMTrainingStep: Codable, Equatable, Sendable {
    public let loss: Double
    public let outputs: HRMEdgeOutputs
    public let nextCarry: HRMCarryState
    public let trace: HRMExecutionTrace

    public init(loss: Double, outputs: HRMEdgeOutputs, nextCarry: HRMCarryState, trace: HRMExecutionTrace) {
        self.loss = loss
        self.outputs = outputs
        self.nextCarry = nextCarry
        self.trace = trace
    }
}

public struct HRMGrowthEvent: Codable, Equatable, Sendable {
    public let dimension: GrowthDimension
    public let previousValue: Int
    public let newValue: Int
    public let anchorBlockLayout: String
    public let activePowers: [Int]
    public let tensorPolicies: [String: GrowthGeometryPolicy]

    public init(
        dimension: GrowthDimension,
        previousValue: Int,
        newValue: Int,
        anchorBlockLayout: String,
        activePowers: [Int],
        tensorPolicies: [String: GrowthGeometryPolicy]
    ) {
        self.dimension = dimension
        self.previousValue = previousValue
        self.newValue = newValue
        self.anchorBlockLayout = anchorBlockLayout
        self.activePowers = activePowers
        self.tensorPolicies = tensorPolicies
    }
}

public struct HRMGrowthMetadata: Codable, Equatable, Sendable {
    public let anchorBlockLayout: String
    public let oneDimensionalFallback: String
    public let bridgeDefaultEnabled: Bool
    public var activePowers: [Int]
    public var history: [HRMGrowthEvent]

    public init(
        anchorBlockLayout: String = "top-left-0-degree-experience",
        oneDimensionalFallback: String = "0,180,180,0",
        bridgeDefaultEnabled: Bool = false,
        activePowers: [Int],
        history: [HRMGrowthEvent] = []
    ) {
        self.anchorBlockLayout = anchorBlockLayout
        self.oneDimensionalFallback = oneDimensionalFallback
        self.bridgeDefaultEnabled = bridgeDefaultEnabled
        self.activePowers = activePowers
        self.history = history
    }
}

public struct HRMModel: HRMEdgeBackend {
    public var edgeNames: [String]
    public var hDim: Int
    public var zDim: Int
    public var yDepth: Int
    public var xPixels: Int
    public var curvature: Double
    public var learningRate: Double
    public var predictionDepth: Int
    public var hLayers: Int
    public var lLayers: Int
    public var hCycles: Int
    public var lCycles: Int
    public private(set) var growthMetadata: HRMGrowthMetadata

    private var predictors: [String: HRMEdgePredictor]
    private var fisheyeCache: [String: FisheyeBuffer]
    private var closeBuffer: [String: [Double]]
    private var edgeCarry: [String: HRMCarryState?]
    private var maxHistory: Int
    private var predictionQueue: [String: [PredictionFrame]]
    private var lastTrace: [String: HRMExecutionTrace]
    private var edgeTimelineCursors: [String: EdgeTimelineCursor]
    private var cachedFisheyeKey: FisheyeCacheKey?
    private var cachedFisheyeBoundaries: [Int]

    private struct PredictionFrame: Codable {
        let fisheye: [Double]
        let barIdx: Int
        let carry: HRMCarryState?
    }

    private struct EdgeTimelineCursor {
        var candleIndex: Int
        var lastBarIdx: Int
    }

    private struct FisheyeCacheKey: Equatable {
        let yDepth: Int
        let xPixels: Int
        let curvature: Double
    }

    private struct ModelState: Codable {
        let hDim: Int
        let zDim: Int
        let yDepth: Int
        let xPixels: Int
        let curvature: Double
        let learningRate: Double
        let predictionDepth: Int
        let hLayers: Int
        let lLayers: Int
        let hCycles: Int
        let lCycles: Int
        let edgeNames: [String]
        let growthMetadata: HRMGrowthMetadata
        let predictors: [String: HRMEdgePredictor.Snapshot]
        let edgeCarry: [String: HRMCarryState]
    }

    public init(
        nEdges: Int = 0,
        hDim: Int = 4,
        zDim: Int = 4,
        yDepth: Int = 200,
        xPixels: Int = 20,
        curvature: Double = 2.0,
        learningRate: Double = 0.001,
        predictionDepth: Int = 1,
        hLayers: Int = 2,
        lLayers: Int = 2,
        hCycles: Int = 2,
        lCycles: Int = 2
    ) {
        precondition(isPowerOf4(hDim), "hDim must be power of 4 (4, 16, 64, 256)")
        precondition(isPowerOf4(zDim), "zDim must be power of 4 (4, 16, 64, 256)")
        precondition(allowedGrowthPowers.contains(hDim), "hDim must be one of 4, 16, 64, 256")
        precondition(allowedGrowthPowers.contains(zDim), "zDim must be one of 4, 16, 64, 256")
        precondition(uniquePowerSet(hDim: hDim, zDim: zDim).count <= 2, "At most two powers of 4 may be active")

        self.edgeNames = []
        self.hDim = hDim
        self.zDim = zDim
        self.yDepth = yDepth
        self.xPixels = xPixels
        self.curvature = curvature
        self.learningRate = learningRate
        self.predictionDepth = predictionDepth
        self.hLayers = hLayers
        self.lLayers = lLayers
        self.hCycles = hCycles
        self.lCycles = lCycles
        self.growthMetadata = HRMGrowthMetadata(activePowers: uniquePowerSet(hDim: hDim, zDim: zDim))
        self.predictors = [:]
        self.fisheyeCache = [:]
        self.closeBuffer = [:]
        self.edgeCarry = [:]
        self.maxHistory = max(yDepth + 100, 500)
        self.predictionQueue = [:]
        self.lastTrace = [:]
        self.edgeTimelineCursors = [:]
        self.cachedFisheyeKey = nil
        self.cachedFisheyeBoundaries = []
        _ = nEdges
    }

    public mutating func registerEdges(_ edges: [String]) {
        edgeNames = edges
        var nextPredictors: [String: HRMEdgePredictor] = [:]

        for edge in edges {
            if let existing = predictors[edge],
               existing.matchesShape(
                    hDim: hDim,
                    zDim: zDim,
                    xPixels: xPixels,
                    hLayers: hLayers,
                    lLayers: lLayers,
                    hCycles: hCycles,
                    lCycles: lCycles
               ) {
                nextPredictors[edge] = existing
            } else {
                nextPredictors[edge] = HRMEdgePredictor(
                    hDim: hDim,
                    zDim: zDim,
                    xPixels: xPixels,
                    curvature: curvature,
                    learningRate: learningRate,
                    hLayers: hLayers,
                    lLayers: lLayers,
                    hCycles: hCycles,
                    lCycles: lCycles
                )
            }

            fisheyeCache[edge] = FisheyeBuffer(yDepth: yDepth, xPixels: xPixels)
            closeBuffer[edge, default: []] = closeBuffer[edge, default: []]
            edgeCarry[edge] = edgeCarry[edge] ?? nil
            predictionQueue[edge] = predictionQueue[edge] ?? []
            edgeTimelineCursors[edge] = edgeTimelineCursors[edge] ?? EdgeTimelineCursor(candleIndex: 0, lastBarIdx: -1)
        }

        predictors = nextPredictors
        invalidateFisheyeBoundaryCache()
    }

    public func getFisheye(edge: String) -> [[Double]] {
        fisheyeCache[edge]?.data ?? []
    }

    public func currentCarry(edge: String) -> HRMCarryState? {
        edgeCarry[edge] ?? nil
    }

    public func trace(edge: String) -> HRMExecutionTrace? {
        lastTrace[edge]
    }

    public func predictorSnapshot(edge: String) -> HRMEdgePredictor.Snapshot? {
        predictors[edge]?.snapshot
    }

    public mutating func updateFisheye(edge: String, value: Double, x: Int, y: Int) {
        ensureEdgeRegistered(edge)
        if var buffer = fisheyeCache[edge] {
            buffer.update(value: value, x: x, y: y)
            fisheyeCache[edge] = buffer
        }
    }

    public mutating func predict(
        commonTimestamps: [Date],
        edges: [String: [DBCandle]],
        barIdx: Int
    ) -> [String: (fraction: Double, ptt: Double, stop: Double)] {
        updatePrices(commonTimestamps: commonTimestamps, edges: edges, barIdx: barIdx)

        var results: [String: (fraction: Double, ptt: Double, stop: Double)] = [:]
        for edge in edgeNames {
            let fisheye = getFisheyeColumn(edge: edge)
            let normalizedFisheye = normalizedFisheyeColumn(fisheye)
            let inputCarry = edgeCarry[edge] ?? nil
            let step = predictNormalized(edge: edge, normalizedFisheye: normalizedFisheye, carry: inputCarry)
            results[edge] = (
                fraction: step.outputs.fraction,
                ptt: step.outputs.ptt,
                stop: step.outputs.stop
            )
            predictionQueue[edge, default: []].append(
                PredictionFrame(fisheye: normalizedFisheye, barIdx: barIdx, carry: inputCarry)
            )
            if predictionQueue[edge]!.count > predictionDepth {
                predictionQueue[edge] = Array(predictionQueue[edge]!.suffix(predictionDepth))
            }
        }
        return results
    }

    public mutating func update(
        commonTimestamps: [Date],
        edges: [String: [DBCandle]],
        edgeAccels: [String: Double],
        barIdx: Int,
        hitPtt: [String: Bool],
        hitStop: [String: Bool]
    ) -> Double? {
        guard !edgeNames.isEmpty else { return nil }
        _ = edgeAccels

        updatePrices(commonTimestamps: commonTimestamps, edges: edges, barIdx: barIdx)

        var totalLoss = 0.0
        var scored = 0

        for edge in edgeNames {
            guard let queue = predictionQueue[edge], queue.count >= predictionDepth else {
                continue
            }

            let frame = queue[0]
            let targets = HRMTargets(
                fraction: hitPtt[edge] == true ? 1.0 : (hitStop[edge] == true ? 0.0 : 0.5),
                ptt: hitPtt[edge] == true ? 1.0 : 0.0,
                stop: hitStop[edge] == true ? 1.0 : 0.0
            )

            guard var predictor = predictors[edge] else {
                continue
            }
            let step = trainNormalized(
                edge: edge,
                predictor: &predictor,
                normalizedFisheye: frame.fisheye,
                targets: targets,
                carry: frame.carry
            )
            totalLoss += step.loss
            scored += 1
        }

        return scored > 0 ? totalLoss / Double(scored) : nil
    }

    public mutating func predict(edge: String, fisheye: [Double], carry: HRMCarryState? = nil) -> HRMPredictionStep {
        let normalizedFisheye = normalizedFisheyeColumn(fisheye)
        return predictNormalized(edge: edge, normalizedFisheye: normalizedFisheye, carry: carry)
    }

    public mutating func train(edge: String, fisheye: [Double], targets: HRMTargets, carry: HRMCarryState? = nil) -> HRMTrainingStep {
        ensureEdgeRegistered(edge)
        var predictor = predictors[edge]!
        let normalizedFisheye = normalizedFisheyeColumn(fisheye)
        return trainNormalized(edge: edge, predictor: &predictor, normalizedFisheye: normalizedFisheye, targets: targets, carry: carry)
    }

    public mutating func predictEdge(
        edge: String,
        fisheyeColumn: [Double]
    ) -> (fraction: Double, ptt: Double, stop: Double) {
        let step = predict(edge: edge, fisheye: fisheyeColumn, carry: nil)
        return (
            fraction: step.outputs.fraction,
            ptt: step.outputs.ptt,
            stop: step.outputs.stop
        )
    }

    public mutating func trainEdge(
        edge: String,
        fisheyeColumn: [Double],
        targetFrac: Double,
        targetPtt: Double,
        targetStop: Double
    ) -> Double? {
        let step = train(
            edge: edge,
            fisheye: fisheyeColumn,
            targets: HRMTargets(fraction: targetFrac, ptt: targetPtt, stop: targetStop),
            carry: nil
        )
        return step.loss
    }

    public mutating func grow(dim: String) {
        guard let dimension = GrowthDimension(rawValue: dim) else {
            return
        }
        grow(dim: dimension, geometryPolicy: nil)
    }

    public mutating func grow(dim: GrowthDimension, geometryPolicy: GrowthGeometryPolicy? = nil) {
        let previousValue: Int
        let newValue: Int

        switch dim {
        case .h:
            guard let next = nextGrowthPower(after: hDim) else { return }
            previousValue = hDim
            hDim = next
            newValue = next
        case .z:
            guard let next = nextGrowthPower(after: zDim) else { return }
            previousValue = zDim
            zDim = next
            newValue = next
        case .H:
            previousValue = hLayers
            hLayers += 1
            newValue = hLayers
        case .L:
            previousValue = lLayers
            lLayers += 1
            newValue = lLayers
        }

        let activePowers = uniquePowerSet(hDim: hDim, zDim: zDim)
        precondition(activePowers.count <= 2, "At most two powers of 4 may be active")
        growthMetadata.activePowers = activePowers

        var tensorPolicies: [String: GrowthGeometryPolicy] = [:]
        for edge in edgeNames {
            guard var predictor = predictors[edge] else { continue }
            let grownPolicies = predictor.grow(
                dimension: dim,
                geometryPolicy: geometryPolicy,
                newHDim: hDim,
                newZDim: zDim
            )
            predictors[edge] = predictor
            let fittedCarry = (edgeCarry[edge] ?? nil)?.fitted(
                hDim: hDim,
                zDim: zDim,
                fallbackH: predictor.initialCarry().zH,
                fallbackL: predictor.initialCarry().zL
            )
            edgeCarry[edge] = fittedCarry
            for (tensor, policy) in grownPolicies {
                tensorPolicies["\(edge).\(tensor)"] = policy
            }
        }

        growthMetadata.history.append(
            HRMGrowthEvent(
                dimension: dim,
                previousValue: previousValue,
                newValue: newValue,
                anchorBlockLayout: growthMetadata.anchorBlockLayout,
                activePowers: activePowers,
                tensorPolicies: tensorPolicies
            )
        )
    }

    public func save(path: String) throws {
        let state = ModelState(
            hDim: hDim,
            zDim: zDim,
            yDepth: yDepth,
            xPixels: xPixels,
            curvature: curvature,
            learningRate: learningRate,
            predictionDepth: predictionDepth,
            hLayers: hLayers,
            lLayers: lLayers,
            hCycles: hCycles,
            lCycles: lCycles,
            edgeNames: edgeNames,
            growthMetadata: growthMetadata,
            predictors: predictors.mapValues { $0.snapshot },
            edgeCarry: edgeCarry.compactMapValues { $0 }
        )

        let data: Data
        do {
            data = try JSONEncoder().encode(state)
        } catch {
            throw HRMCheckpointError.encodeFailed(path: path, reason: error.localizedDescription)
        }

        do {
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            throw HRMCheckpointError.writeFailed(path: path, reason: error.localizedDescription)
        }
    }

    public mutating func load(path: String) throws {
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: path))
        } catch {
            let nsError = error as NSError
            if nsError.domain == NSCocoaErrorDomain,
               nsError.code == CocoaError.fileReadNoSuchFile.rawValue {
                throw HRMCheckpointError.fileMissing(path: path)
            }
            throw HRMCheckpointError.readFailed(path: path, reason: error.localizedDescription)
        }

        let state: ModelState
        do {
            state = try JSONDecoder().decode(ModelState.self, from: data)
        } catch {
            throw HRMCheckpointError.decodeFailed(path: path, reason: error.localizedDescription)
        }

        hDim = state.hDim
        zDim = state.zDim
        yDepth = state.yDepth
        xPixels = state.xPixels
        curvature = state.curvature
        learningRate = state.learningRate
        predictionDepth = state.predictionDepth
        hLayers = state.hLayers
        lLayers = state.lLayers
        hCycles = state.hCycles
        lCycles = state.lCycles
        growthMetadata = state.growthMetadata
        maxHistory = max(yDepth + 100, 500)

        let targetEdges = edgeNames.isEmpty ? state.edgeNames : edgeNames
        edgeNames = targetEdges
        predictors = [:]
        fisheyeCache = [:]
        predictionQueue = [:]
        lastTrace = [:]
        edgeTimelineCursors = [:]
        cachedFisheyeKey = nil
        cachedFisheyeBoundaries = []

        for edge in targetEdges {
            if let snapshot = state.predictors[edge] {
                predictors[edge] = HRMEdgePredictor(snapshot: snapshot)
            } else {
                predictors[edge] = HRMEdgePredictor(
                    hDim: hDim,
                    zDim: zDim,
                    xPixels: xPixels,
                    curvature: curvature,
                    learningRate: learningRate,
                    hLayers: hLayers,
                    lLayers: lLayers,
                    hCycles: hCycles,
                    lCycles: lCycles
                )
            }

            fisheyeCache[edge] = FisheyeBuffer(yDepth: yDepth, xPixels: xPixels)
            closeBuffer[edge, default: []] = closeBuffer[edge, default: []]
            predictionQueue[edge] = []
            edgeTimelineCursors[edge] = EdgeTimelineCursor(candleIndex: 0, lastBarIdx: -1)

            let predictor = predictors[edge]!
            edgeCarry[edge] = state.edgeCarry[edge]?.fitted(
                hDim: hDim,
                zDim: zDim,
                fallbackH: predictor.initialCarry().zH,
                fallbackL: predictor.initialCarry().zL
            )
        }
    }

    private mutating func ensureEdgeRegistered(_ edge: String) {
        if !edgeNames.contains(edge) {
            edgeNames.append(edge)
        }

        if predictors[edge] == nil {
            predictors[edge] = HRMEdgePredictor(
                hDim: hDim,
                zDim: zDim,
                xPixels: xPixels,
                curvature: curvature,
                learningRate: learningRate,
                hLayers: hLayers,
                lLayers: lLayers,
                hCycles: hCycles,
                lCycles: lCycles
            )
        }

        if fisheyeCache[edge] == nil {
            fisheyeCache[edge] = FisheyeBuffer(yDepth: yDepth, xPixels: xPixels)
        }
        closeBuffer[edge, default: []] = closeBuffer[edge, default: []]
        predictionQueue[edge, default: []] = predictionQueue[edge, default: []]
        edgeCarry[edge] = edgeCarry[edge] ?? nil
        edgeTimelineCursors[edge] = edgeTimelineCursors[edge] ?? EdgeTimelineCursor(candleIndex: 0, lastBarIdx: -1)
    }

    private mutating func updatePrices(commonTimestamps: [Date], edges: [String: [DBCandle]], barIdx: Int) {
        guard barIdx >= 0, barIdx < commonTimestamps.count else { return }
        let timestamp = commonTimestamps[barIdx]

        for edge in edgeNames {
            guard let candles = edges[edge], !candles.isEmpty else {
                continue
            }

            var cursor = edgeTimelineCursors[edge] ?? EdgeTimelineCursor(candleIndex: 0, lastBarIdx: -1)
            if barIdx < cursor.lastBarIdx || cursor.candleIndex >= candles.count {
                cursor.candleIndex = 0
            }
            if barIdx == cursor.lastBarIdx {
                edgeTimelineCursors[edge] = cursor
                continue
            }

            while cursor.candleIndex < candles.count && candles[cursor.candleIndex].timestamp < timestamp {
                cursor.candleIndex += 1
            }

            guard cursor.candleIndex < candles.count,
                  candles[cursor.candleIndex].timestamp == timestamp else {
                edgeTimelineCursors[edge] = cursor
                continue
            }

            let candle = candles[cursor.candleIndex]
            var history = closeBuffer[edge, default: []]
            history.append(candle.close)
            if history.count > maxHistory {
                history.removeFirst(history.count - maxHistory)
            }
            closeBuffer[edge] = history
            cursor.lastBarIdx = barIdx
            edgeTimelineCursors[edge] = cursor
        }
    }

    private mutating func getFisheyeColumn(edge: String) -> [Double] {
        let closes = closeBuffer[edge] ?? []
        guard closes.count >= 2 else {
            return Array(repeating: 0.0, count: xPixels)
        }

        let recent = closes.suffix(yDepth)
        let boundaries = currentFisheyeBoundaries()
        return fisheyeSample(candles: recent, boundaries: boundaries, xPixels: xPixels)
    }

    private mutating func cacheFisheye(edge: String, fisheye: [Double]) {
        guard var buffer = fisheyeCache[edge] else { return }
        buffer.pushRow(values: fisheye)
        fisheyeCache[edge] = buffer
    }

    private mutating func predictNormalized(
        edge: String,
        normalizedFisheye: [Double],
        carry: HRMCarryState?
    ) -> HRMPredictionStep {
        ensureEdgeRegistered(edge)
        let safeCarry = (carry ?? edgeCarry[edge] ?? nil) ?? predictors[edge]!.initialCarry()
        let step = predictors[edge]!.forwardStep(fisheyeColumn: normalizedFisheye, carry: safeCarry)
        edgeCarry[edge] = step.nextCarry
        lastTrace[edge] = step.trace
        cacheFisheye(edge: edge, fisheye: normalizedFisheye)
        return step
    }

    private mutating func trainNormalized(
        edge: String,
        predictor: inout HRMEdgePredictor,
        normalizedFisheye: [Double],
        targets: HRMTargets,
        carry: HRMCarryState?
    ) -> HRMTrainingStep {
        ensureEdgeRegistered(edge)
        let safeCarry = (carry ?? edgeCarry[edge] ?? nil) ?? predictor.initialCarry()
        let step = predictor.trainStep(
            fisheyeColumn: normalizedFisheye,
            targets: targets,
            carry: safeCarry
        )
        predictors[edge] = predictor
        edgeCarry[edge] = step.nextCarry
        lastTrace[edge] = step.trace
        return step
    }

    private func normalizedFisheyeColumn(_ values: [Double]) -> [Double] {
        guard values.count != xPixels else { return values }
        return fitVector(values, count: xPixels)
    }

    private mutating func currentFisheyeBoundaries() -> [Int] {
        let key = FisheyeCacheKey(yDepth: yDepth, xPixels: xPixels, curvature: curvature)
        if cachedFisheyeKey != key {
            cachedFisheyeKey = key
            cachedFisheyeBoundaries = fisheyeBoundaries(yDepth: yDepth, xPixels: xPixels, curvature: curvature)
        }
        return cachedFisheyeBoundaries
    }

    private mutating func invalidateFisheyeBoundaryCache() {
        cachedFisheyeKey = nil
        cachedFisheyeBoundaries = []
    }
}

public struct FisheyeBuffer {
    public var yDepth: Int
    public var xPixels: Int
    private var storage: [[Double]]
    private var head: Int

    public var data: [[Double]] {
        guard !storage.isEmpty else { return [] }

        var rows: [[Double]] = []
        rows.reserveCapacity(storage.count)
        for offset in 0..<storage.count {
            rows.append(storage[(head + offset) % storage.count])
        }
        return rows
    }

    public init(yDepth: Int, xPixels: Int) {
        self.yDepth = yDepth
        self.xPixels = xPixels
        self.storage = Array(
            repeating: Array(repeating: 0.0, count: max(0, xPixels)),
            count: max(0, yDepth)
        )
        self.head = 0
    }

    public mutating func update(value: Double, x: Int, y: Int) {
        guard !storage.isEmpty, y >= 0, y < yDepth, x >= 0, x < xPixels else { return }
        let physicalRow = (head + y) % storage.count
        storage[physicalRow][x] = value
    }

    public mutating func pushRow(values: [Double]) {
        guard !storage.isEmpty else { return }
        let row = values.count == xPixels ? values : fitVector(values, count: xPixels)
        storage[head] = row
        head = (head + 1) % storage.count
    }
}

public struct HRMEdgePredictor {
    public struct Snapshot: Codable, Equatable {
        public let hDim: Int
        public let zDim: Int
        public let xPixels: Int
        public let curvature: Double
        public let learningRate: Double
        public let hLayers: Int
        public let lLayers: Int
        public let hCycles: Int
        public let lCycles: Int
        public let inputProjectionL: [[Double]]
        public let hToL: [[Double]]
        public let lToL: [[Double]]
        public let lLevelLayers: [[[Double]]]
        public let lToH: [[Double]]
        public let hToH: [[Double]]
        public let hLevelLayers: [[[Double]]]
        public let hInit: [Double]
        public let lInit: [Double]
        public let fractionHead: [Double]
        public let pttHead: [Double]
        public let stopHead: [Double]
        public let outputBiases: [Double]
    }

    fileprivate var hDim: Int
    fileprivate var zDim: Int
    fileprivate var xPixels: Int
    fileprivate var curvature: Double
    fileprivate var learningRate: Double
    fileprivate var hLayers: Int
    fileprivate var lLayers: Int
    fileprivate var hCycles: Int
    fileprivate var lCycles: Int

    private var inputProjectionL: [[Double]]
    private var hToL: [[Double]]
    private var lToL: [[Double]]
    private var lLevelLayers: [[[Double]]]
    private var lToH: [[Double]]
    private var hToH: [[Double]]
    private var hLevelLayers: [[[Double]]]
    private var hInit: [Double]
    private var lInit: [Double]
    private var fractionHead: [Double]
    private var pttHead: [Double]
    private var stopHead: [Double]
    private var outputBiases: [Double]

    var snapshot: Snapshot {
        Snapshot(
            hDim: hDim,
            zDim: zDim,
            xPixels: xPixels,
            curvature: curvature,
            learningRate: learningRate,
            hLayers: hLayers,
            lLayers: lLayers,
            hCycles: hCycles,
            lCycles: lCycles,
            inputProjectionL: inputProjectionL,
            hToL: hToL,
            lToL: lToL,
            lLevelLayers: lLevelLayers,
            lToH: lToH,
            hToH: hToH,
            hLevelLayers: hLevelLayers,
            hInit: hInit,
            lInit: lInit,
            fractionHead: fractionHead,
            pttHead: pttHead,
            stopHead: stopHead,
            outputBiases: outputBiases
        )
    }

    init(
        hDim: Int,
        zDim: Int,
        xPixels: Int,
        curvature: Double,
        learningRate: Double,
        hLayers: Int,
        lLayers: Int,
        hCycles: Int,
        lCycles: Int
    ) {
        self.hDim = hDim
        self.zDim = zDim
        self.xPixels = xPixels
        self.curvature = curvature
        self.learningRate = learningRate
        self.hLayers = hLayers
        self.lLayers = lLayers
        self.hCycles = hCycles
        self.lCycles = lCycles
        self.inputProjectionL = randomMatrix(rows: zDim, cols: xPixels, std: 0.02)
        self.hToL = randomMatrix(rows: zDim, cols: hDim, std: 0.02)
        self.lToL = identityMatrix(size: zDim, diagonal: 0.8)
        self.lLevelLayers = (0..<lLayers).map { _ in identityMatrix(size: zDim, diagonal: 1.0) }
        self.lToH = randomMatrix(rows: hDim, cols: zDim, std: 0.02)
        self.hToH = identityMatrix(size: hDim, diagonal: 0.8)
        self.hLevelLayers = (0..<hLayers).map { _ in identityMatrix(size: hDim, diagonal: 1.0) }
        self.hInit = Array(repeating: 0.0, count: hDim)
        self.lInit = Array(repeating: 0.0, count: zDim)
        self.fractionHead = Array(repeating: 0.0, count: hDim)
        self.pttHead = Array(repeating: 0.0, count: hDim)
        self.stopHead = Array(repeating: 0.0, count: hDim)
        self.outputBiases = [0.0, 0.0, 0.0]
    }

    init(snapshot: Snapshot) {
        self.hDim = snapshot.hDim
        self.zDim = snapshot.zDim
        self.xPixels = snapshot.xPixels
        self.curvature = snapshot.curvature
        self.learningRate = snapshot.learningRate
        self.hLayers = snapshot.hLayers
        self.lLayers = snapshot.lLayers
        self.hCycles = snapshot.hCycles
        self.lCycles = snapshot.lCycles
        self.inputProjectionL = snapshot.inputProjectionL
        self.hToL = snapshot.hToL
        self.lToL = snapshot.lToL
        self.lLevelLayers = snapshot.lLevelLayers
        self.lToH = snapshot.lToH
        self.hToH = snapshot.hToH
        self.hLevelLayers = snapshot.hLevelLayers
        self.hInit = snapshot.hInit
        self.lInit = snapshot.lInit
        self.fractionHead = snapshot.fractionHead
        self.pttHead = snapshot.pttHead
        self.stopHead = snapshot.stopHead
        self.outputBiases = snapshot.outputBiases
    }

    func matchesShape(
        hDim: Int,
        zDim: Int,
        xPixels: Int,
        hLayers: Int,
        lLayers: Int,
        hCycles: Int,
        lCycles: Int
    ) -> Bool {
        self.hDim == hDim &&
        self.zDim == zDim &&
        self.xPixels == xPixels &&
        self.hLayers == hLayers &&
        self.lLayers == lLayers &&
        self.hCycles == hCycles &&
        self.lCycles == lCycles
    }

    func initialCarry() -> HRMCarryState {
        HRMCarryState(zH: hInit, zL: lInit)
    }

    mutating func forwardStep(fisheyeColumn: [Double], carry: HRMCarryState) -> HRMPredictionStep {
        let cache = forwardCache(fisheyeColumn: fisheyeColumn, carry: carry)
        return HRMPredictionStep(outputs: cache.outputs, nextCarry: cache.nextCarry, trace: cache.trace)
    }

    mutating func trainStep(
        fisheyeColumn: [Double],
        targets: HRMTargets,
        carry: HRMCarryState
    ) -> HRMTrainingStep {
        let cache = forwardCache(fisheyeColumn: fisheyeColumn, carry: carry)
        let targetsVector = [targets.fraction, targets.ptt, targets.stop]
        let outputVector = [cache.outputs.fraction, cache.outputs.ptt, cache.outputs.stop]
        let oldHeads = [fractionHead, pttHead, stopHead]
        let outputGrads = zip(outputVector, targetsVector).map { $0 - $1 }
        let loss = zip(outputVector, targetsVector).map { bceLoss(pred: $0, target: $1) }.reduce(0.0, +)

        for index in 0..<fractionHead.count {
            fractionHead[index] -= learningRate * outputGrads[0] * cache.finalH[index]
            pttHead[index] -= learningRate * outputGrads[1] * cache.finalH[index]
            stopHead[index] -= learningRate * outputGrads[2] * cache.finalH[index]
        }
        outputBiases[0] -= learningRate * outputGrads[0]
        outputBiases[1] -= learningRate * outputGrads[1]
        outputBiases[2] -= learningRate * outputGrads[2]

        var gradH = [Double](repeating: 0.0, count: hDim)
        for hIdx in 0..<hDim {
            gradH[hIdx] =
                oldHeads[0][hIdx] * outputGrads[0] +
                oldHeads[1][hIdx] * outputGrads[1] +
                oldHeads[2][hIdx] * outputGrads[2]
        }

        let hDerivative = cache.finalH.map { 1.0 - ($0 * $0) }
        gradH = zip(gradH, hDerivative).map(*)

        let oldHToH = hToH
        let oldLToH = lToH
        updateMatrix(matrix: &hToH, grad: outer(gradH, cache.hSourceH), learningRate: learningRate)
        updateMatrix(matrix: &lToH, grad: outer(gradH, cache.hSourceL), learningRate: learningRate)

        var hBack = gradH
        for layerIndex in stride(from: hLevelLayers.count - 1, through: 0, by: -1) where !hLevelLayers.isEmpty {
            let layerInput = cache.hLayerInputs[layerIndex]
            let localGrad = zip(hBack, cache.hLayerOutputs[layerIndex].map { 1.0 - ($0 * $0) }).map(*)
            let oldLayer = hLevelLayers[layerIndex]
            updateMatrix(matrix: &hLevelLayers[layerIndex], grad: outer(localGrad, layerInput), learningRate: learningRate)
            hBack = matvec(A: transpose(oldLayer), x: localGrad)
        }

        var gradL = matvec(A: transpose(oldLToH), x: gradH)
        let lDerivative = cache.finalL.map { 1.0 - ($0 * $0) }
        gradL = zip(gradL, lDerivative).map(*)

        let oldInputProjectionL = inputProjectionL
        let oldHToL = hToL
        let oldLToL = lToL
        updateMatrix(matrix: &inputProjectionL, grad: outer(gradL, cache.x), learningRate: learningRate)
        updateMatrix(matrix: &hToL, grad: outer(gradL, cache.lSourceH), learningRate: learningRate)
        updateMatrix(matrix: &lToL, grad: outer(gradL, cache.lSourceL), learningRate: learningRate)

        var lBack = gradL
        for layerIndex in stride(from: lLevelLayers.count - 1, through: 0, by: -1) where !lLevelLayers.isEmpty {
            let layerInput = cache.lLayerInputs[layerIndex]
            let localGrad = zip(lBack, cache.lLayerOutputs[layerIndex].map { 1.0 - ($0 * $0) }).map(*)
            let oldLayer = lLevelLayers[layerIndex]
            updateMatrix(matrix: &lLevelLayers[layerIndex], grad: outer(localGrad, layerInput), learningRate: learningRate)
            lBack = matvec(A: transpose(oldLayer), x: localGrad)
        }

        let gradInitH = matvec(A: transpose(oldHToH), x: gradH)
        let gradInitL = matvec(A: transpose(oldLToL), x: gradL)
        for index in 0..<hInit.count {
            hInit[index] -= learningRate * gradInitH[index]
        }
        for index in 0..<lInit.count {
            lInit[index] -= learningRate * gradInitL[index]
        }

        _ = oldInputProjectionL
        _ = oldHToL

        return HRMTrainingStep(loss: loss, outputs: cache.outputs, nextCarry: cache.nextCarry, trace: cache.trace)
    }

    mutating func grow(
        dimension: GrowthDimension,
        geometryPolicy: GrowthGeometryPolicy?,
        newHDim: Int,
        newZDim: Int
    ) -> [String: GrowthGeometryPolicy] {
        var tensorPolicies: [String: GrowthGeometryPolicy] = [:]

        switch dimension {
        case .h:
            let squarePolicy = resolveGrowthPolicy(
                override: geometryPolicy,
                rows: hDim,
                cols: hDim,
                growsRows: true,
                growsCols: true
            )
            let axisPolicy = resolveGrowthPolicy(
                override: geometryPolicy,
                rows: zDim,
                cols: hDim,
                growsRows: false,
                growsCols: true
            )

            hToL = growColumns(hToL, policy: axisPolicy)
            lToH = growRows(lToH, policy: axisPolicy)
            hToH = growSquare(hToH, policy: squarePolicy)
            hLevelLayers = hLevelLayers.map { growSquare($0, policy: squarePolicy) }
            hInit = growVector(hInit, policy: axisPolicy)
            fractionHead = growVector(fractionHead, policy: axisPolicy)
            pttHead = growVector(pttHead, policy: axisPolicy)
            stopHead = growVector(stopHead, policy: axisPolicy)

            hDim = newHDim
            tensorPolicies["hToL"] = axisPolicy
            tensorPolicies["lToH"] = axisPolicy
            tensorPolicies["hToH"] = squarePolicy
            tensorPolicies["hLevelLayers"] = squarePolicy
            tensorPolicies["hInit"] = axisPolicy
            tensorPolicies["fractionHead"] = axisPolicy
            tensorPolicies["pttHead"] = axisPolicy
            tensorPolicies["stopHead"] = axisPolicy

        case .z:
            let squarePolicy = resolveGrowthPolicy(
                override: geometryPolicy,
                rows: zDim,
                cols: zDim,
                growsRows: true,
                growsCols: true
            )
            let axisPolicy = resolveGrowthPolicy(
                override: geometryPolicy,
                rows: zDim,
                cols: hDim,
                growsRows: true,
                growsCols: false
            )

            inputProjectionL = growRows(inputProjectionL, policy: axisPolicy)
            hToL = growRows(hToL, policy: axisPolicy)
            lToL = growSquare(lToL, policy: squarePolicy)
            lLevelLayers = lLevelLayers.map { growSquare($0, policy: squarePolicy) }
            lToH = growColumns(lToH, policy: axisPolicy)
            lInit = growVector(lInit, policy: axisPolicy)

            zDim = newZDim
            tensorPolicies["inputProjectionL"] = axisPolicy
            tensorPolicies["hToL"] = axisPolicy
            tensorPolicies["lToL"] = squarePolicy
            tensorPolicies["lLevelLayers"] = squarePolicy
            tensorPolicies["lToH"] = axisPolicy
            tensorPolicies["lInit"] = axisPolicy

        case .H:
            hLevelLayers.append(identityMatrix(size: hDim, diagonal: 1.0))
            hLayers += 1

        case .L:
            lLevelLayers.append(identityMatrix(size: zDim, diagonal: 1.0))
            lLayers += 1
        }

        return tensorPolicies
    }

    private func forwardCache(fisheyeColumn: [Double], carry: HRMCarryState) -> ForwardCache {
        let x = fitVector(fisheyeColumn, count: xPixels)
        let safeCarry = carry.fitted(hDim: hDim, zDim: zDim, fallbackH: hInit, fallbackL: lInit)
        let inputL = tanhVector(matvec(A: inputProjectionL, x: x))

        var zH = safeCarry.zH
        var zL = safeCarry.zL
        var lSourceH = zH
        var lSourceL = zL
        var hSourceH = zH
        var hSourceL = zL
        var lLayerInputs: [[Double]] = Array(repeating: [], count: max(0, lLevelLayers.count))
        var lLayerOutputs: [[Double]] = Array(repeating: [], count: max(0, lLevelLayers.count))
        var hLayerInputs: [[Double]] = Array(repeating: [], count: max(0, hLevelLayers.count))
        var hLayerOutputs: [[Double]] = Array(repeating: [], count: max(0, hLevelLayers.count))
        var cycleOrder: [String] = []

        for hPass in 0..<hCycles {
            for lPass in 0..<lCycles {
                cycleOrder.append("L\(hPass).\(lPass)")
                lSourceH = zH
                lSourceL = zL
                var lState = tanhVector(
                    addVectors([
                        inputL,
                        matvec(A: hToL, x: zH),
                        matvec(A: lToL, x: zL)
                    ])
                )

                for index in 0..<lLevelLayers.count {
                    lLayerInputs[index] = lState
                    lState = tanhVector(
                        addVectors([
                            matvec(A: lLevelLayers[index], x: lState),
                            inputL
                        ])
                    )
                    lLayerOutputs[index] = lState
                }

                zL = lState
            }

            cycleOrder.append("H\(hPass)")
            hSourceH = zH
            hSourceL = zL
            let injectedL = matvec(A: lToH, x: zL)
            var hState = tanhVector(
                addVectors([
                    injectedL,
                    matvec(A: hToH, x: zH)
                ])
            )

            for index in 0..<hLevelLayers.count {
                hLayerInputs[index] = hState
                hState = tanhVector(
                    addVectors([
                        matvec(A: hLevelLayers[index], x: hState),
                        injectedL
                    ])
                )
                hLayerOutputs[index] = hState
            }

            zH = hState
        }

        let logits = [
            dot(fractionHead, zH) + outputBiases[0],
            dot(pttHead, zH) + outputBiases[1],
            dot(stopHead, zH) + outputBiases[2]
        ]
        let outputs = HRMEdgeOutputs(
            fraction: sigmoid(logits[0]),
            ptt: sigmoid(logits[1]),
            stop: sigmoid(logits[2])
        )

        return ForwardCache(
            x: x,
            finalH: zH,
            finalL: zL,
            hSourceH: hSourceH,
            hSourceL: hSourceL,
            lSourceH: lSourceH,
            lSourceL: lSourceL,
            hLayerInputs: hLayerInputs,
            hLayerOutputs: hLayerOutputs,
            lLayerInputs: lLayerInputs,
            lLayerOutputs: lLayerOutputs,
            outputs: outputs,
            nextCarry: HRMCarryState(zH: zH, zL: zL),
            trace: HRMExecutionTrace(hPasses: hCycles, lPasses: hCycles * lCycles, cycleOrder: cycleOrder)
        )
    }

    private struct ForwardCache {
        let x: [Double]
        let finalH: [Double]
        let finalL: [Double]
        let hSourceH: [Double]
        let hSourceL: [Double]
        let lSourceH: [Double]
        let lSourceL: [Double]
        let hLayerInputs: [[Double]]
        let hLayerOutputs: [[Double]]
        let lLayerInputs: [[Double]]
        let lLayerOutputs: [[Double]]
        let outputs: HRMEdgeOutputs
        let nextCarry: HRMCarryState
        let trace: HRMExecutionTrace
    }
}

public func fisheyeBoundaries(yDepth: Int, xPixels: Int, curvature: Double) -> [Int] {
    guard xPixels > 1 else { return [yDepth] }

    var boundaries: [Int] = []
    var previous = 0
    for index in 0..<xPixels {
        let t = Double(index) / Double(xPixels - 1)
        let warped = pow(t, curvature)
        let boundary = max(Int(Double(yDepth) * warped), previous + 1)
        boundaries.append(boundary)
        previous = boundary
    }
    return boundaries
}

public func fisheyeSample<C: RandomAccessCollection>(candles: C, boundaries: [Int], xPixels: Int) -> [Double]
where C.Element == Double {
    guard !candles.isEmpty else {
        return Array(repeating: 0.0, count: xPixels)
    }

    let candleCount = candles.count
    var prefixSums = [Double](repeating: 0.0, count: candleCount + 1)
    var runningSum = 0.0
    var position = 0
    for candle in candles {
        runningSum += candle
        position += 1
        prefixSums[position] = runningSum
    }

    var results: [Double] = []
    results.reserveCapacity(xPixels)
    var previous = 0
    let current = candles[candles.index(candles.startIndex, offsetBy: candleCount - 1)]

    for boundary in boundaries.prefix(xPixels) {
        let end = min(boundary, candleCount)
        if end <= previous {
            results.append(0.0)
        } else {
            let mean = (prefixSums[end] - prefixSums[previous]) / Double(end - previous)
            results.append(mean == 0.0 ? 0.0 : (current - mean) / mean)
        }
        previous = boundary
    }

    if results.count < xPixels {
        results.append(contentsOf: repeatElement(0.0, count: xPixels - results.count))
    }
    return results
}

public func randomMatrix(rows: Int, cols: Int, std: Double) -> [[Double]] {
    (0..<rows).map { _ in
        (0..<cols).map { _ in Double.random(in: -std ... std) }
    }
}

public func zeroMatrix(rows: Int, cols: Int) -> [[Double]] {
    Array(repeating: Array(repeating: 0.0, count: cols), count: rows)
}

public func identityMatrix(size: Int, diagonal: Double = 1.0) -> [[Double]] {
    var matrix = zeroMatrix(rows: size, cols: size)
    for index in 0..<size {
        matrix[index][index] = diagonal
    }
    return matrix
}

public func matvec(A: [[Double]], x: [Double]) -> [Double] {
    guard !A.isEmpty else { return [] }
    precondition(A[0].count == x.count, "Dimension mismatch: rows=\(A.count) cols=\(A[0].count) x=\(x.count)")

    var result = [Double](repeating: 0.0, count: A.count)
    for row in 0..<A.count {
        var total = 0.0
        for col in 0..<x.count {
            total += A[row][col] * x[col]
        }
        result[row] = total
    }
    return result
}

public func transpose(_ matrix: [[Double]]) -> [[Double]] {
    guard let cols = matrix.first?.count else { return [] }
    var result = zeroMatrix(rows: cols, cols: matrix.count)
    for row in 0..<matrix.count {
        for col in 0..<cols {
            result[col][row] = matrix[row][col]
        }
    }
    return result
}

public func outer(_ a: [Double], _ b: [Double]) -> [[Double]] {
    var result: [[Double]] = []
    result.reserveCapacity(a.count)
    for lhs in a {
        var row: [Double] = []
        row.reserveCapacity(b.count)
        for rhs in b {
            row.append(lhs * rhs)
        }
        result.append(row)
    }
    return result
}

public func dot(_ a: [Double], _ b: [Double]) -> Double {
    precondition(a.count == b.count, "Vector length mismatch")
    var total = 0.0
    for index in 0..<a.count {
        total += a[index] * b[index]
    }
    return total
}

public func sigmoid(_ x: Double) -> Double {
    1.0 / (1.0 + exp(-x))
}

public func bceLoss(pred: Double, target: Double) -> Double {
    let eps = 1e-7
    let p = max(min(pred, 1.0 - eps), eps)
    return -(target * log(p) + (1.0 - target) * log(1.0 - p))
}

public func isPowerOf4(_ n: Int) -> Bool {
    guard n > 0 else { return false }
    if (n & (n - 1)) != 0 { return false }

    var value = n
    while value % 4 == 0 {
        value /= 4
    }
    return value == 1
}

public func topLeftBlock(_ matrix: [[Double]], size: Int) -> [[Double]] {
    Array(matrix.prefix(size)).map { Array($0.prefix(size)) }
}

public func uniquePowerSet(hDim: Int, zDim: Int) -> [Int] {
    Array(Set([hDim, zDim])).sorted()
}

public func nextGrowthPower(after current: Int) -> Int? {
    guard let index = allowedGrowthPowers.firstIndex(of: current), index + 1 < allowedGrowthPowers.count else {
        return nil
    }
    return allowedGrowthPowers[index + 1]
}

public func resolveGrowthPolicy(
    override: GrowthGeometryPolicy?,
    rows: Int,
    cols: Int,
    growsRows: Bool,
    growsCols: Bool
) -> GrowthGeometryPolicy {
    if let override {
        return override
    }

    if growsRows && growsCols && rows == cols {
        return .quadrantRotation
    }

    return .axialFallback0_180_180_0
}

private func fitVector(_ values: [Double], count: Int, fallback: [Double]? = nil) -> [Double] {
    if fallback == nil && values.count == count {
        return values
    }
    var result = Array(values.prefix(count))
    if result.count < count {
        let filler = fallback ?? Array(repeating: 0.0, count: count)
        result.append(contentsOf: filler.dropFirst(result.count).prefix(count - result.count))
    }
    return result
}

private func tanhVector(_ values: [Double]) -> [Double] {
    values.map { tanh($0) }
}

private func addVectors(_ vectors: [[Double]]) -> [Double] {
    guard let count = vectors.first?.count else { return [] }
    var result = Array(repeating: 0.0, count: count)
    for vector in vectors {
        precondition(vector.count == count, "Vector size mismatch")
        for index in 0..<count {
            result[index] += vector[index]
        }
    }
    return result
}

private func updateMatrix(matrix: inout [[Double]], grad: [[Double]], learningRate: Double) {
    guard matrix.count == grad.count else { return }
    for row in 0..<matrix.count {
        guard matrix[row].count == grad[row].count else { continue }
        for col in 0..<matrix[row].count {
            matrix[row][col] -= learningRate * grad[row][col]
        }
    }
}

private func rotate180(_ matrix: [[Double]]) -> [[Double]] {
    matrix.reversed().map { Array($0.reversed()) }
}

private func rotate90Square(_ matrix: [[Double]]) -> [[Double]] {
    let n = matrix.count
    var result = zeroMatrix(rows: n, cols: n)
    for row in 0..<n {
        for col in 0..<n {
            result[n - 1 - col][row] = matrix[row][col]
        }
    }
    return result
}

private func rotate270Square(_ matrix: [[Double]]) -> [[Double]] {
    let n = matrix.count
    var result = zeroMatrix(rows: n, cols: n)
    for row in 0..<n {
        for col in 0..<n {
            result[col][n - 1 - row] = matrix[row][col]
        }
    }
    return result
}

private func growVector(_ vector: [Double], policy: GrowthGeometryPolicy) -> [Double] {
    switch policy {
    case .quadrantRotation, .axialFallback0_180_180_0:
        let reversed = Array(vector.reversed())
        return vector + reversed + reversed + vector
    case .randomRestart:
        return vector +
            randomVector(count: vector.count, std: 0.02) +
            randomVector(count: vector.count, std: 0.02) +
            randomVector(count: vector.count, std: 0.02)
    case .bridgeCompatibility:
        return vector + Array(repeating: 0.0, count: vector.count * 3)
    }
}

private func growRows(_ matrix: [[Double]], policy: GrowthGeometryPolicy) -> [[Double]] {
    switch policy {
    case .quadrantRotation, .axialFallback0_180_180_0:
        let rotated = rotate180(matrix)
        return matrix + rotated + rotated + matrix
    case .randomRestart:
        return matrix +
            randomMatrix(rows: matrix.count, cols: matrix.first?.count ?? 0, std: 0.02) +
            randomMatrix(rows: matrix.count, cols: matrix.first?.count ?? 0, std: 0.02) +
            randomMatrix(rows: matrix.count, cols: matrix.first?.count ?? 0, std: 0.02)
    case .bridgeCompatibility:
        return matrix + zeroMatrix(rows: matrix.count * 3, cols: matrix.first?.count ?? 0)
    }
}

private func growColumns(_ matrix: [[Double]], policy: GrowthGeometryPolicy) -> [[Double]] {
    guard !matrix.isEmpty else { return matrix }

    switch policy {
    case .quadrantRotation, .axialFallback0_180_180_0:
        let rotated = rotate180(matrix)
        return zip4Horizontal(matrix, rotated, rotated, matrix)
    case .randomRestart:
        return zip4Horizontal(
            matrix,
            randomMatrix(rows: matrix.count, cols: matrix[0].count, std: 0.02),
            randomMatrix(rows: matrix.count, cols: matrix[0].count, std: 0.02),
            randomMatrix(rows: matrix.count, cols: matrix[0].count, std: 0.02)
        )
    case .bridgeCompatibility:
        return zip4Horizontal(
            matrix,
            zeroMatrix(rows: matrix.count, cols: matrix[0].count),
            zeroMatrix(rows: matrix.count, cols: matrix[0].count),
            zeroMatrix(rows: matrix.count, cols: matrix[0].count)
        )
    }
}

private func growSquare(_ matrix: [[Double]], policy: GrowthGeometryPolicy) -> [[Double]] {
    guard !matrix.isEmpty else { return matrix }
    let blockSize = matrix.count

    switch policy {
    case .quadrantRotation:
        let variants = [matrix, rotate180(matrix), rotate90Square(matrix), rotate270Square(matrix)]
        let layout = [
            [0, 1, 2, 3],
            [2, 3, 0, 1],
            [1, 0, 3, 2],
            [3, 2, 1, 0]
        ]
        var result = zeroMatrix(rows: blockSize * 4, cols: blockSize * 4)
        for blockRow in 0..<4 {
            for blockCol in 0..<4 {
                let block = variants[layout[blockRow][blockCol]]
                for row in 0..<blockSize {
                    for col in 0..<blockSize {
                        result[blockRow * blockSize + row][blockCol * blockSize + col] = block[row][col]
                    }
                }
            }
        }
        return result

    case .axialFallback0_180_180_0:
        let variants = [matrix, rotate180(matrix), rotate180(matrix), matrix]
        var result = zeroMatrix(rows: blockSize * 4, cols: blockSize * 4)
        for blockRow in 0..<4 {
            for blockCol in 0..<4 {
                let block = variants[max(blockRow, blockCol)]
                for row in 0..<blockSize {
                    for col in 0..<blockSize {
                        result[blockRow * blockSize + row][blockCol * blockSize + col] = block[row][col]
                    }
                }
            }
        }
        return result

    case .randomRestart:
        var result = randomMatrix(rows: blockSize * 4, cols: blockSize * 4, std: 0.02)
        for row in 0..<blockSize {
            for col in 0..<blockSize {
                result[row][col] = matrix[row][col]
            }
        }
        return result

    case .bridgeCompatibility:
        var result = zeroMatrix(rows: blockSize * 4, cols: blockSize * 4)
        for row in 0..<blockSize {
            for col in 0..<blockSize {
                result[row][col] = matrix[row][col]
            }
        }
        return result
    }
}

private func zip4Horizontal(
    _ a: [[Double]],
    _ b: [[Double]],
    _ c: [[Double]],
    _ d: [[Double]]
) -> [[Double]] {
    var result = zeroMatrix(rows: a.count, cols: (a.first?.count ?? 0) * 4)
    for row in 0..<a.count {
        result[row] = a[row] + b[row] + c[row] + d[row]
    }
    return result
}

private func randomVector(count: Int, std: Double) -> [Double] {
    (0..<count).map { _ in Double.random(in: -std ... std) }
}
