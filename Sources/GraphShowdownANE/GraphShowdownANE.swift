//
//  GraphShowdownANE.swift
//  GraphShowdownANE
//
//  Shared backend contract for GraphShowdown training.
//  Uses a real Espresso ANE recurrent core when available and falls back to CPU HRM otherwise.
//

import Foundation
import AutotradeHRM
import Espresso

public struct ANETrainingConfig {
    public let dim: Int
    public let hidden: Int
    public let seqLen: Int
    public let xPixels: Int
    public let curvature: Float
    public let heads: Int
    public let layers: Int
    public let learningRate: Float

    public init(
        dim: Int = 4,
        hidden: Int = 4,
        seqLen: Int = 16,
        xPixels: Int = 20,
        curvature: Float = 2.0,
        heads: Int = 1,
        layers: Int = 2,
        learningRate: Float = 0.001
    ) {
        self.dim = dim
        self.hidden = hidden
        self.seqLen = seqLen
        self.xPixels = xPixels
        self.curvature = curvature
        self.heads = heads
        self.layers = layers
        self.learningRate = learningRate
    }
}

public enum ANEBackendMode: String, Codable, Sendable {
    case ane
    case cpuFallback
}

public struct ANEExecutionProofMetrics: Codable, Equatable, Sendable {
    public let aneMs: Double
    public let ioMs: Double

    public init(aneMs: Double, ioMs: Double) {
        self.aneMs = aneMs
        self.ioMs = ioMs
    }
}

public struct ANEProofSummary: Codable, Equatable, Sendable {
    public let backendMode: ANEBackendMode
    public let aneStepCount: Int
    public let aneMsTotal: Double
    public let ioMsTotal: Double

    public init(
        backendMode: ANEBackendMode,
        aneStepCount: Int,
        aneMsTotal: Double,
        ioMsTotal: Double
    ) {
        self.backendMode = backendMode
        self.aneStepCount = aneStepCount
        self.aneMsTotal = aneMsTotal
        self.ioMsTotal = ioMsTotal
    }

    public var hasProof: Bool {
        aneStepCount > 0 && aneMsTotal > 0
    }
}

let aneInputTileWidth = 8

struct ANEInputLayout {
    let logicalWidth: Int
    let tileWidth: Int
    let paddedWidth: Int
    let tileCount: Int

    init(logicalWidth: Int, tileWidth: Int = aneInputTileWidth) {
        let boundedLogicalWidth = max(1, logicalWidth)
        let boundedTileWidth = max(1, tileWidth)
        self.logicalWidth = boundedLogicalWidth
        self.tileWidth = boundedTileWidth
        self.paddedWidth = ((boundedLogicalWidth + boundedTileWidth - 1) / boundedTileWidth) * boundedTileWidth
        self.tileCount = paddedWidth / boundedTileWidth
    }
}

private struct ANEInputProjectionBank {
    let rowCount: Int
    let layout: ANEInputLayout

    private let weights: [Float]

    init(rowCount: Int, layout: ANEInputLayout, seed: UInt64 = 0xD00DFEED1234ABCD) {
        self.rowCount = rowCount
        self.layout = layout

        var generator = ProjectionLCG(seed: seed)
        var generated: [Float] = []
        generated.reserveCapacity(rowCount * layout.paddedWidth)
        for _ in 0..<(rowCount * layout.paddedWidth) {
            generated.append((generator.nextUnitFloat() * 2 - 1) * 0.01)
        }
        self.weights = generated
    }

    func project(_ input: [Float]) -> [Float] {
        precondition(input.count == layout.paddedWidth, "Packed ANE input width mismatch")

        let paddedWidth = layout.paddedWidth
        var projected = [Float](repeating: 0.0, count: rowCount)

        for row in 0..<rowCount {
            let rowOffset = row * paddedWidth
            var total: Float = 0.0
            for tile in 0..<layout.tileCount {
                let tileBase = tile * layout.tileWidth
                let tileLimit = tileBase + layout.tileWidth
                for col in tileBase..<tileLimit {
                    total += weights[rowOffset + col] * input[col]
                }
            }
            projected[row] = total
        }

        return projected
    }

    private struct ProjectionLCG {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state = state &* 2862933555777941757 &+ 3037000493
            return state
        }

        mutating func nextUnitFloat() -> Float {
            Float(next() & 0x00FF_FFFF) / Float(0x0100_0000)
        }
    }
}

struct ANECoreStepResult {
    let output: [Float]
    let metrics: ANEExecutionProofMetrics
}

protocol ANECoreRunning {
    func runSingleStep(input: [Float], resetState: Bool) throws -> ANECoreStepResult
}

public enum ANECheckpointBoundaryError: Error, Equatable, LocalizedError {
    case saveUnavailable(path: String, reason: String)
    case loadUnavailable(path: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case let .saveUnavailable(path, reason):
            return "ANE checkpoint save unavailable for \(path): \(reason)"
        case let .loadUnavailable(path, reason):
            return "ANE checkpoint load unavailable for \(path): \(reason)"
        }
    }
}

private struct RealANECore: ANECoreRunning {
    let core: AutotradeANERecurrentCore

    func runSingleStep(input: [Float], resetState: Bool) throws -> ANECoreStepResult {
        let result = try core.runSingleStep(input: input, resetState: resetState)
        return ANECoreStepResult(
            output: result.output,
            metrics: ANEExecutionProofMetrics(
                aneMs: result.metrics.aneMs,
                ioMs: result.metrics.ioMs
            )
        )
    }
}

private final class ANEProofLedger {
    private(set) var stepCount: Int = 0
    private(set) var aneMsTotal: Double = 0.0
    private(set) var ioMsTotal: Double = 0.0

    func record(_ metrics: ANEExecutionProofMetrics) {
        stepCount += 1
        aneMsTotal += metrics.aneMs
        ioMsTotal += metrics.ioMs
    }
}

private final class ANEEdgeState {
    let edge: String
    let xPixels: Int
    let hDim: Int
    let zDim: Int
    let learningRate: Float
    private(set) var fractionHead: [Float]
    private(set) var pttHead: [Float]
    private(set) var stopHead: [Float]
    private(set) var biases: [Float]

    init(edge: String, xPixels: Int, hDim: Int, zDim: Int, learningRate: Float) {
        self.edge = edge
        self.xPixels = xPixels
        self.hDim = hDim
        self.zDim = zDim
        self.learningRate = learningRate

        let seed = ANEEdgeState.seedForEdge(edge)
        self.fractionHead = ANEEdgeState.makeHead(seed: seed &+ 1)
        self.pttHead = ANEEdgeState.makeHead(seed: seed &+ 2)
        self.stopHead = ANEEdgeState.makeHead(seed: seed &+ 3)
        self.biases = [0.0, 0.0, 0.0]
    }

    func predict(from hidden: [Float], metrics: ANEExecutionProofMetrics) -> HRMPredictionStep {
        let outputs = forwardOutputs(hidden: hidden)
        return HRMPredictionStep(
            outputs: outputs,
            nextCarry: makeCarry(hidden: hidden),
            trace: HRMExecutionTrace(
                hPasses: 1,
                lPasses: 1,
                cycleOrder: [
                    "ANE.recurrent",
                    String(format: "ANE.ms=%.3f", metrics.aneMs),
                    String(format: "IO.ms=%.3f", metrics.ioMs)
                ]
            )
        )
    }

    func train(from hidden: [Float], targets: HRMTargets, metrics: ANEExecutionProofMetrics) -> HRMTrainingStep {
        let outputs = forwardOutputs(hidden: hidden)
        let outputVector = [outputs.fraction, outputs.ptt, outputs.stop]
        let targetVector = [targets.fraction, targets.ptt, targets.stop]
        let grads = zip(outputVector, targetVector).map { Float($0 - $1) }

        let loss = zip(outputVector, targetVector).map { bceLoss(pred: $0, target: $1) }.reduce(0.0, +)

        for index in 0..<hidden.count {
            fractionHead[index] -= learningRate * grads[0] * hidden[index]
            pttHead[index] -= learningRate * grads[1] * hidden[index]
            stopHead[index] -= learningRate * grads[2] * hidden[index]
        }
        biases[0] -= learningRate * grads[0]
        biases[1] -= learningRate * grads[1]
        biases[2] -= learningRate * grads[2]

        return HRMTrainingStep(
            loss: loss,
            outputs: outputs,
            nextCarry: makeCarry(hidden: hidden),
            trace: HRMExecutionTrace(
                hPasses: 1,
                lPasses: 1,
                cycleOrder: [
                    "ANE.recurrent",
                    "CPU.headUpdate",
                    String(format: "ANE.ms=%.3f", metrics.aneMs)
                ]
            )
        )
    }

    private func forwardOutputs(hidden: [Float]) -> HRMEdgeOutputs {
        HRMEdgeOutputs(
            fraction: Double(sigmoidFloat(dot(fractionHead, hidden) + biases[0])),
            ptt: Double(sigmoidFloat(dot(pttHead, hidden) + biases[1])),
            stop: Double(sigmoidFloat(dot(stopHead, hidden) + biases[2]))
        )
    }

    private func makeCarry(hidden: [Float]) -> HRMCarryState {
        let hCarry = hidden.prefix(hDim).map(Double.init)
        let zCarry = hidden.suffix(zDim).map(Double.init)
        return HRMCarryState(zH: hCarry, zL: zCarry)
    }

    private static func makeHead(seed: UInt64) -> [Float] {
        var rng = SeededLCG(seed: seed)
        return (0..<AutotradeANERecurrentCore.dim).map { _ in (rng.nextUnitFloat() * 2 - 1) * 0.005 }
    }

    private static func seedForEdge(_ edge: String) -> UInt64 {
        var hash: UInt64 = 1469598103934665603
        for byte in edge.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return hash
    }

    private struct SeededLCG {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
        }

        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }

        mutating func nextUnitFloat() -> Float {
            Float(next() & 0x00FF_FFFF) / Float(0x0100_0000)
        }
    }
}

public final class ANEModelWrapper: HRMEdgeBackend {
    public let config: ANETrainingConfig
    public private(set) var backendMode: ANEBackendMode

    private var hrmModel: HRMModel
    private var aneCore: ANECoreRunning?
    private let inputLayout: ANEInputLayout
    private var inputProjection: ANEInputProjectionBank
    private var edgeStates: [String: ANEEdgeState]
    private let proofLedger = ANEProofLedger()

    public var proofSummary: ANEProofSummary {
        ANEProofSummary(
            backendMode: backendMode,
            aneStepCount: proofLedger.stepCount,
            aneMsTotal: proofLedger.aneMsTotal,
            ioMsTotal: proofLedger.ioMsTotal
        )
    }

    public init(config: ANETrainingConfig) throws {
        let inputLayout = ANEInputLayout(logicalWidth: config.xPixels)
        self.config = config
        self.hrmModel = HRMModel(
            hDim: config.dim,
            zDim: config.hidden,
            yDepth: config.seqLen,
            xPixels: config.xPixels,
            curvature: Double(config.curvature),
            learningRate: Double(config.learningRate),
            predictionDepth: 1,
            hLayers: config.layers,
            lLayers: config.layers
        )
        self.inputLayout = inputLayout
        self.inputProjection = ANEInputProjectionBank(
            rowCount: AutotradeANERecurrentCore.dim,
            layout: inputLayout
        )
        self.edgeStates = [:]

        do {
            self.aneCore = RealANECore(core: try AutotradeANERecurrentCore(seed: 0xA17E20260326))
            self.backendMode = .ane
        } catch {
            self.aneCore = nil
            self.backendMode = .cpuFallback
        }
    }

    public convenience init(model: HRMModel, backendMode: ANEBackendMode = .cpuFallback) {
        self.init(model: model, backendMode: backendMode, core: nil)
    }

    init(model: HRMModel, backendMode: ANEBackendMode = .cpuFallback, core: ANECoreRunning?) {
        let inputLayout = ANEInputLayout(logicalWidth: model.xPixels)
        self.config = ANETrainingConfig(
            dim: model.hDim,
            hidden: model.zDim,
            seqLen: model.yDepth,
            xPixels: model.xPixels,
            curvature: Float(model.curvature),
            heads: 1,
            layers: max(model.hLayers, model.lLayers),
            learningRate: Float(model.learningRate)
        )
        self.hrmModel = model
        self.inputLayout = inputLayout
        self.inputProjection = ANEInputProjectionBank(
            rowCount: AutotradeANERecurrentCore.dim,
            layout: inputLayout
        )
        self.edgeStates = [:]
        self.backendMode = backendMode
        self.aneCore = core
    }

    public func registerEdges(_ edges: [String]) {
        hrmModel.registerEdges(edges)

        guard backendMode == .ane else {
            return
        }

        var nextStates: [String: ANEEdgeState] = [:]
        for edge in edges {
            nextStates[edge] = edgeStates[edge] ?? ANEEdgeState(
                edge: edge,
                xPixels: config.xPixels,
                hDim: config.dim,
                zDim: config.hidden,
                learningRate: config.learningRate
            )
        }
        edgeStates = nextStates
    }

    public func predict(edge: String, fisheye: [Double], carry: HRMCarryState?) -> HRMPredictionStep {
        guard backendMode == .ane,
              let core = aneCore else {
            return hrmModel.predict(edge: edge, fisheye: fisheye, carry: carry)
        }

        let state = ensureEdgeState(edge: edge)
        do {
            let projected = projectInput(fisheye, carry: carry)
            let result = try core.runSingleStep(input: projected, resetState: true)
            proofLedger.record(result.metrics)
            return state.predict(from: result.output, metrics: result.metrics)
        } catch {
            backendMode = .cpuFallback
            return hrmModel.predict(edge: edge, fisheye: fisheye, carry: carry)
        }
    }

    public func train(edge: String, fisheye: [Double], targets: HRMTargets, carry: HRMCarryState?) -> HRMTrainingStep {
        guard backendMode == .ane,
              let core = aneCore else {
            return hrmModel.train(edge: edge, fisheye: fisheye, targets: targets, carry: carry)
        }

        let state = ensureEdgeState(edge: edge)
        do {
            let projected = projectInput(fisheye, carry: carry)
            let result = try core.runSingleStep(input: projected, resetState: true)
            proofLedger.record(result.metrics)
            return state.train(from: result.output, targets: targets, metrics: result.metrics)
        } catch {
            backendMode = .cpuFallback
            return hrmModel.train(edge: edge, fisheye: fisheye, targets: targets, carry: carry)
        }
    }

    public func grow(dim: GrowthDimension, geometryPolicy: GrowthGeometryPolicy?) {
        hrmModel.grow(dim: dim, geometryPolicy: geometryPolicy)
    }

    public func forward(edge: String, fisheye: [Double], carry: HRMCarryState? = nil) throws -> (frac: Float, ptt: Float, stop: Float, nextCarry: HRMCarryState) {
        let step = predict(edge: edge, fisheye: fisheye, carry: carry)
        return (
            frac: Float(step.outputs.fraction),
            ptt: Float(step.outputs.ptt),
            stop: Float(step.outputs.stop),
            nextCarry: step.nextCarry
        )
    }

    public func trainStep(
        edge: String,
        fisheye: [Double],
        targets: HRMTargets,
        carry: HRMCarryState? = nil
    ) throws -> (loss: Float, nextCarry: HRMCarryState) {
        let step = train(edge: edge, fisheye: fisheye, targets: targets, carry: carry)
        return (loss: Float(step.loss), nextCarry: step.nextCarry)
    }

    public func save(path: String) throws {
        guard backendMode == .ane else {
            try hrmModel.save(path: path)
            return
        }

        throw ANECheckpointBoundaryError.saveUnavailable(
            path: path,
            reason: "Active ANE wrapper state has no truthful checkpoint format in this repo."
        )
    }

    public func load(path: String) throws {
        guard backendMode == .ane else {
            try hrmModel.load(path: path)
            return
        }

        throw ANECheckpointBoundaryError.loadUnavailable(
            path: path,
            reason: "Active ANE wrapper state has no truthful checkpoint format in this repo."
        )
    }

    private func ensureEdgeState(edge: String) -> ANEEdgeState {
        if let existing = edgeStates[edge] {
            return existing
        }

        let state = ANEEdgeState(
            edge: edge,
            xPixels: config.xPixels,
            hDim: config.dim,
            zDim: config.hidden,
            learningRate: config.learningRate
        )
        edgeStates[edge] = state
        return state
    }

    private func projectInput(_ fisheye: [Double], carry: HRMCarryState?) -> [Float] {
        let input = packANEInputRow(
            fisheye,
            layout: inputLayout,
            curvature: Double(config.curvature)
        )
        var projected = inputProjection.project(input)

        if let carry {
            let fitted = carry.fitted(
                hDim: config.dim,
                zDim: config.hidden,
                fallbackH: Array(repeating: 0.0, count: config.dim),
                fallbackL: Array(repeating: 0.0, count: config.hidden)
            )

            for (index, value) in fitted.zH.enumerated() where index < projected.count {
                projected[index] += Float(value)
            }

            let zOffset = min(AutotradeANERecurrentCore.dim / 2, projected.count)
            for (index, value) in fitted.zL.enumerated() where zOffset + index < projected.count {
                projected[zOffset + index] += Float(value)
            }
        }

        return projected
    }
}

func packANEInputRow(_ fisheye: [Double], layout: ANEInputLayout, curvature: Double = 2.0) -> [Float] {
    let logicalInput: [Double]
    if fisheye.count > layout.logicalWidth {
        let boundaries = fisheyeBoundaries(
            yDepth: fisheye.count,
            xPixels: layout.logicalWidth,
            curvature: curvature
        )
        logicalInput = fisheyeSample(
            candles: fisheye,
            boundaries: boundaries,
            xPixels: layout.logicalWidth
        )
    } else if fisheye.count == layout.logicalWidth {
        logicalInput = fisheye
    } else {
        logicalInput = fisheye + Array(repeating: 0.0, count: layout.logicalWidth - fisheye.count)
    }

    var packed = [Float](repeating: 0.0, count: layout.paddedWidth)
    for index in 0..<layout.logicalWidth {
        packed[index] = Float(logicalInput[index])
    }
    return packed
}

private func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
    precondition(lhs.count == rhs.count)
    var total: Float = 0.0
    for index in 0..<lhs.count {
        total += lhs[index] * rhs[index]
    }
    return total
}

private func sigmoidFloat(_ x: Float) -> Float {
    1.0 / (1.0 + exp(-x))
}
