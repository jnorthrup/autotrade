//
//  IntegrationTests.swift
//  AutotradeTests
//

import Foundation
import Testing
@testable import Autotrade
@testable import AutotradeHRM
@testable import GraphShowdownANE
import Espresso

private func makeANEProofGraph() -> CoinGraph {
    let graph = CoinGraph(feeRate: 0.001)
    let start = Date(timeIntervalSince1970: 1_700_000_000)
    let next = start.addingTimeInterval(60)
    let candles = [
        DBCandle(timestamp: start, open: 100.0, high: 101.0, low: 99.0, close: 100.5, volume: 10.0),
        DBCandle(timestamp: next, open: 100.5, high: 102.0, low: 100.0, close: 101.5, volume: 12.0)
    ]

    graph.setEdges(
        edges: ["BTC-USD": candles],
        edgeState: ["BTC-USD": EdgeState()],
        nodeState: ["BTC": NodeState(), "USD": NodeState()],
        nodes: ["BTC", "USD"],
        allPairs: ["BTC-USD"],
        commonTimestamps: [start, next]
    )
    return graph
}

final class ANEStubCore: ANECoreRunning {
    private(set) var callCount: Int = 0
    private let result: ANECoreStepResult

    init(result: ANECoreStepResult) {
        self.result = result
    }

    func runSingleStep(input: [Float], resetState: Bool) throws -> ANECoreStepResult {
        callCount += 1
        return result
    }
}

@Suite("Integration Tests")
struct IntegrationTests {
    @Test("HRM model edge registration")
    func testFullPipeline() {
        var model = HRMModel(
            nEdges: 5,
            hDim: 4,
            zDim: 4,
            learningRate: 0.001
        )

        let testEdges = ["BTC-USD", "ETH-USD"]
        model.registerEdges(testEdges)
        #expect(model.edgeNames == testEdges)
    }

    @Test("Performance baseline")
    func testPerformanceBaseline() {
        let start = Date()

        var model = HRMModel(
            nEdges: 10,
            hDim: 16,
            zDim: 16,
            yDepth: 50,
            xPixels: 10
        )

        let edges = (0..<10).map { "PAIR-\($0)" }
        model.registerEdges(edges)

        let duration = Date().timeIntervalSince(start)
        #expect(duration < 1.0)
    }

    @Test("ANE wrapper shares CPU checkpoint contract in fallback mode")
    func testANEContractParity() throws {
        let checkpoint = FileManager.default.temporaryDirectory.appendingPathComponent("ane_parity.json").path

        var cpu = HRMModel(hDim: 4, zDim: 4, yDepth: 16, xPixels: 4)
        cpu.registerEdges(["BTC-USD"])
        _ = cpu.predict(edge: "BTC-USD", fisheye: [0.1, 0.2, -0.3, 0.4])
        try cpu.save(path: checkpoint)

        let ane = ANEModelWrapper(
            model: HRMModel(hDim: 4, zDim: 4, yDepth: 16, xPixels: 4),
            backendMode: .cpuFallback
        )
        ane.registerEdges(["BTC-USD"])
        try ane.load(path: checkpoint)
        print("ANE backend mode: \(ane.backendMode.rawValue)")

        let cpuStep = cpu.predict(edge: "BTC-USD", fisheye: [0.1, 0.2, -0.3, 0.4])
        let aneStep = ane.predict(edge: "BTC-USD", fisheye: [0.1, 0.2, -0.3, 0.4], carry: nil)

        #expect(ane.backendMode == .cpuFallback)
        #expect(abs(cpuStep.outputs.fraction - aneStep.outputs.fraction) < 1e-12)
        #expect(abs(cpuStep.outputs.ptt - aneStep.outputs.ptt) < 1e-12)
        #expect(abs(cpuStep.outputs.stop - aneStep.outputs.stop) < 1e-12)
        #expect(cpuStep.nextCarry == aneStep.nextCarry)
    }

    @Test("ANE wrapper fails closed on checkpoint persistence while ANE is active")
    func testANECheckpointPersistenceFailsClosedWhenANEActive() {
        let checkpoint = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-live-\(UUID().uuidString).json")
            .path
        let hidden = Array(repeating: Float(0.25), count: AutotradeANERecurrentCore.dim)
        let stubCore = ANEStubCore(
            result: ANECoreStepResult(
                output: hidden,
                metrics: ANEExecutionProofMetrics(aneMs: 1.0, ioMs: 0.5)
            )
        )
        let ane = ANEModelWrapper(
            model: HRMModel(hDim: 4, zDim: 4, yDepth: 16, xPixels: 4),
            backendMode: .ane,
            core: stubCore
        )
        ane.registerEdges(["BTC-USD"])

        do {
            try ane.save(path: checkpoint)
            #expect(Bool(false), "Expected save to fail closed while ANE is active")
        } catch let error as ANECheckpointBoundaryError {
            #expect(
                error
                    == .saveUnavailable(
                        path: checkpoint,
                        reason: "Active ANE wrapper state has no truthful checkpoint format in this repo."
                    )
            )
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }

        do {
            try ane.load(path: checkpoint)
            #expect(Bool(false), "Expected load to fail closed while ANE is active")
        } catch let error as ANECheckpointBoundaryError {
            #expect(
                error
                    == .loadUnavailable(
                        path: checkpoint,
                        reason: "Active ANE wrapper state has no truthful checkpoint format in this repo."
                    )
            )
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }
    }

    @Test("ANE wrapper train contract preserves schema")
    func testANETrainContract() throws {
        let ane = try ANEModelWrapper(config: ANETrainingConfig(dim: 4, hidden: 4, seqLen: 16, xPixels: 4))
        ane.registerEdges(["ETH-USD"])
        print("ANE backend mode (train): \(ane.backendMode.rawValue)")

        let step = ane.train(
            edge: "ETH-USD",
            fisheye: [0.2, 0.1, 0.0, -0.1],
            targets: HRMTargets(fraction: 1.0, ptt: 0.0, stop: 0.0),
            carry: nil
        )

        #expect(step.loss.isFinite)
        #expect(step.nextCarry.zH.count == 4)
        #expect(step.nextCarry.zL.count == 4)
        if ane.backendMode == .ane {
            #expect(step.trace.cycleOrder.first == "ANE.recurrent")
        }
    }

    @Test("ANE HRM trainer fails closed on training and checkpoints")
    func testHRMANETrainerFailsClosed() {
        var trainer = HRMANETrainer()
        let model = HRMModel(hDim: 4, zDim: 4, yDepth: 16, xPixels: 4)
        let edge = "BTC-USD"
        let fisheyeData = [
            [0.1, 0.2],
            [0.3, 0.4]
        ]
        let targets = [(frac: 1.0, ptt: 0.0, stop: 0.0)]
        let exportPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-export-\(UUID().uuidString).ckpt")
            .path
        let importPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-import-\(UUID().uuidString).ckpt")
            .path

        do {
            _ = try trainer.trainEdge(
                edge: edge,
                fisheyeData: fisheyeData,
                targets: targets,
                steps: 7
            )
            #expect(Bool(false), "Expected trainEdge to throw")
        } catch let error as ANEError {
            #expect(
                error
                    == .trainingUnavailable(
                        edge: edge,
                        fisheyeRows: 2,
                        fisheyeColumns: 2,
                        targetCount: 1,
                        steps: 7,
                        reason: "HRMANETrainer is a placeholder; Espresso-backed ANE training is not implemented in this repo."
                    )
            )
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }

        do {
            try trainer.exportWeights(model: model, to: exportPath)
            #expect(Bool(false), "Expected exportWeights to throw")
        } catch let error as ANEError {
            #expect(
                error
                    == .checkpointExportUnavailable(
                        path: exportPath,
                        reason: "Checkpoint export is not implemented for the HRM ANE trainer."
                    )
            )
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }

        do {
            _ = try trainer.importWeights(from: importPath)
            #expect(Bool(false), "Expected importWeights to throw")
        } catch let error as ANEError {
            #expect(
                error
                    == .checkpointImportUnavailable(
                        path: importPath,
                        reason: "Checkpoint import is not implemented for the HRM ANE trainer."
                    )
            )
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }
    }

    @Test("ANE input packing pads to tile width")
    func testANETwitterPacking() {
        let layout = ANEInputLayout(logicalWidth: 10, tileWidth: 8)
        let packed = packANEInputRow((1...10).map(Double.init), layout: layout)

        #expect(layout.paddedWidth == 16)
        #expect(layout.tileCount == 2)
        #expect(packed.count == 16)
        #expect(packed[0] == 1.0)
        #expect(packed[9] == 10.0)
        for index in 10..<packed.count {
            #expect(packed[index] == 0.0)
        }
    }

    @Test("ANE fallback remains explicit when not required")
    func testANENotRequiredAllowsFallback() throws {
        let graph = CoinGraph(feeRate: 0.001)
        let config = ANETrainingConfig(dim: 4, hidden: 4, seqLen: 16, xPixels: 4)

        let result = try runANETraining(
            graph: graph,
            config: config,
            aneRequired: false,
            makeModel: { cfg in
                ANEModelWrapper(
                    model: HRMModel(
                        hDim: cfg.dim,
                        zDim: cfg.hidden,
                        yDepth: cfg.seqLen,
                        xPixels: cfg.xPixels,
                        curvature: Double(cfg.curvature),
                        learningRate: Double(cfg.learningRate),
                        predictionDepth: 1,
                        hLayers: cfg.layers,
                        lLayers: cfg.layers
                    ),
                    backendMode: .cpuFallback
                )
            }
        )

        #expect(result.nUpdates == 0)
        #expect(result.aneProof == nil || result.aneProof?.hasProof == false)
    }

    @Test("ANE required mode fails closed on fallback backend")
    func testANERequiredFailsClosedOnFallbackBackend() {
        let graph = CoinGraph(feeRate: 0.001)
        let config = ANETrainingConfig(dim: 4, hidden: 4, seqLen: 16, xPixels: 4)

        do {
            _ = try runANETraining(
                graph: graph,
                config: config,
                aneRequired: true,
                makeModel: { cfg in
                    ANEModelWrapper(
                        model: HRMModel(
                            hDim: cfg.dim,
                            zDim: cfg.hidden,
                            yDepth: cfg.seqLen,
                            xPixels: cfg.xPixels,
                            curvature: Double(cfg.curvature),
                            learningRate: Double(cfg.learningRate),
                            predictionDepth: 1,
                            hLayers: cfg.layers,
                            lLayers: cfg.layers
                        ),
                        backendMode: .cpuFallback
                    )
                }
            )
            #expect(Bool(false), "Expected ANE required mode to throw on fallback backend")
        } catch let error as ANETrainingError {
            #expect(error == .backendUnavailable)
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }
    }

    @Test("ANE proof summary records real ANE work")
    func testANEProofSummaryIsRecorded() throws {
        let graph = makeANEProofGraph()
        let config = ANETrainingConfig(dim: 4, hidden: 4, seqLen: 16, xPixels: 4)
        let hidden = Array(repeating: Float(0.25), count: AutotradeANERecurrentCore.dim)
        let stubCore = ANEStubCore(
            result: ANECoreStepResult(
                output: hidden,
                metrics: ANEExecutionProofMetrics(aneMs: 2.5, ioMs: 1.5)
            )
        )

        let result = try runANETraining(
            graph: graph,
            config: config,
            aneRequired: true,
            makeModel: { cfg in
                ANEModelWrapper(
                    model: HRMModel(
                        hDim: cfg.dim,
                        zDim: cfg.hidden,
                        yDepth: cfg.seqLen,
                        xPixels: cfg.xPixels,
                        curvature: Double(cfg.curvature),
                        learningRate: Double(cfg.learningRate),
                        predictionDepth: 1,
                        hLayers: cfg.layers,
                        lLayers: cfg.layers
                    ),
                    backendMode: .ane,
                    core: stubCore
                )
            }
        )

        #expect(result.nUpdates > 0)
        #expect(result.aneProof?.hasProof == true)
        #expect(result.aneProof?.aneStepCount == stubCore.callCount)
        #expect(result.aneProof?.aneMsTotal == 5.0)
        #expect(result.aneProof?.ioMsTotal == 3.0)
    }

    @Test("ANE required mode fails closed when proof counters stay zero")
    func testANERequiredFailsClosedOnZeroProof() {
        let graph = makeANEProofGraph()
        let config = ANETrainingConfig(dim: 4, hidden: 4, seqLen: 16, xPixels: 4)
        let hidden = Array(repeating: Float(0.25), count: AutotradeANERecurrentCore.dim)
        let stubCore = ANEStubCore(
            result: ANECoreStepResult(
                output: hidden,
                metrics: ANEExecutionProofMetrics(aneMs: 0.0, ioMs: 0.0)
            )
        )

        do {
            _ = try runANETraining(
                graph: graph,
                config: config,
                aneRequired: true,
                makeModel: { cfg in
                    ANEModelWrapper(
                        model: HRMModel(
                            hDim: cfg.dim,
                            zDim: cfg.hidden,
                            yDepth: cfg.seqLen,
                            xPixels: cfg.xPixels,
                            curvature: Double(cfg.curvature),
                            learningRate: Double(cfg.learningRate),
                            predictionDepth: 1,
                            hLayers: cfg.layers,
                            lLayers: cfg.layers
                        ),
                        backendMode: .ane,
                        core: stubCore
                    )
                }
            )
            #expect(Bool(false), "Expected ANE required mode to throw on zero proof")
        } catch let error as ANETrainingError {
            switch error {
            case .noProof(let proof):
                #expect(proof.aneStepCount == stubCore.callCount)
                #expect(proof.aneMsTotal == 0.0)
                #expect(proof.ioMsTotal == 0.0)
            default:
                #expect(Bool(false), "Unexpected error: \(error)")
            }
        } catch {
            #expect(Bool(false), "Unexpected error: \(error)")
        }
    }
}
