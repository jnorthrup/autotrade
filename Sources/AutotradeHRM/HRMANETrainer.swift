//
//  HRMANETrainer.swift
//  AutotradeHRM
//
//  Integrate Espresso ANE training for HRM model.
//  Uses Espresso's ANE infrastructure for forward/backward passes.
//

import Foundation

// ANE training for HRM using Espresso infrastructure
public struct HRMANETrainer {
    public let hDim: Int
    public let zDim: Int
    public var learningRate: Double

    public init(hDim: Int = 4, zDim: Int = 4, learningRate: Double = 0.001) {
        precondition(hDim == zDim, "Square invariant: hDim must equal zDim")
        self.hDim = hDim
        self.zDim = zDim
        self.learningRate = learningRate
    }

    /// Train HRM edge predictor using ANE.
    ///
    /// This surface is fail-closed until the real Espresso-backed trainer exists.
    /// - Parameters:
    ///   - edge: Edge identifier
    ///   - fisheyeData: Fisheye features [yDepth, xPixels]
    ///   - targets: Target values (frac, ptt, stop)
    ///   - steps: Number of training steps
    /// - Throws: `ANEError.trainingUnavailable` until the real trainer exists
    public mutating func trainEdge(
        edge: String,
        fisheyeData: [[Double]],
        targets: [(frac: Double, ptt: Double, stop: Double)],
        steps: Int = 100
    ) throws -> Double {
        throw ANEError.trainingUnavailable(
            edge: edge,
            fisheyeRows: fisheyeData.count,
            fisheyeColumns: fisheyeData.first?.count ?? 0,
            targetCount: targets.count,
            steps: steps,
            reason: "HRMANETrainer is a placeholder; Espresso-backed ANE training is not implemented in this repo."
        )
    }

    /// Export HRM weights to Espresso-compatible format
    public func exportWeights(model: HRMModel, to path: String) throws {
        _ = model
        throw ANEError.checkpointExportUnavailable(
            path: path,
            reason: "Checkpoint export is not implemented for the HRM ANE trainer."
        )
    }

    /// Import HRM weights from Espresso checkpoint
    public mutating func importWeights(from path: String) throws -> HRMModel {
        throw ANEError.checkpointImportUnavailable(
            path: path,
            reason: "Checkpoint import is not implemented for the HRM ANE trainer."
        )
    }
}

enum ANEError: Error, Equatable, LocalizedError {
    case trainingUnavailable(
        edge: String,
        fisheyeRows: Int,
        fisheyeColumns: Int,
        targetCount: Int,
        steps: Int,
        reason: String
    )
    case checkpointExportUnavailable(path: String, reason: String)
    case checkpointImportUnavailable(path: String, reason: String)

    var errorDescription: String? {
        switch self {
        case let .trainingUnavailable(edge, fisheyeRows, fisheyeColumns, targetCount, steps, reason):
            return "ANE training unavailable for edge \(edge) (fisheye=\(fisheyeRows)x\(fisheyeColumns), samples=\(targetCount), steps=\(steps)): \(reason)"
        case let .checkpointExportUnavailable(path, reason):
            return "ANE checkpoint export unavailable for \(path): \(reason)"
        case let .checkpointImportUnavailable(path, reason):
            return "ANE checkpoint import unavailable for \(path): \(reason)"
        }
    }
}
