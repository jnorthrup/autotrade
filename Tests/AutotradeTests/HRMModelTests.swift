//
//  HRMModelTests.swift
//  AutotradeTests
//

import Foundation
import Testing
@testable import AutotradeHRM

@Suite("HRM Model Tests")
struct HRMModelTests {
    @Test("HRMModel initialization")
    func testHRMModelInitialization() {
        let model = HRMModel(
            nEdges: 10,
            hDim: 4,
            zDim: 4,
            yDepth: 100,
            xPixels: 10,
            curvature: 2.0,
            learningRate: 0.001
        )

        #expect(model.hDim == 4)
        #expect(model.zDim == 4)
        #expect(model.yDepth == 100)
        #expect(model.xPixels == 10)
        #expect(model.curvature == 2.0)
        #expect(model.learningRate == 0.001)
        #expect(model.edgeNames.isEmpty)
        #expect(model.growthMetadata.anchorBlockLayout == "top-left-0-degree-experience")
    }

    @Test("Power-of-4 dimension enforcement")
    func testPowerOf4Dimensions() {
        #expect(isPowerOf4(4))
        #expect(isPowerOf4(16))
        #expect(!isPowerOf4(8))
        #expect(uniquePowerSet(hDim: 4, zDim: 16) == [4, 16])
    }

    @Test("Carry state round trip")
    func testCarryRoundTrip() {
        var model = HRMModel(hDim: 4, zDim: 4, xPixels: 4, hLayers: 2, lLayers: 2, hCycles: 2, lCycles: 3)
        model.registerEdges(["BTC-USD"])

        let fisheye = [0.1, -0.2, 0.3, 0.4]
        let prediction = model.predict(edge: "BTC-USD", fisheye: fisheye, carry: nil)
        let training = model.train(
            edge: "BTC-USD",
            fisheye: fisheye,
            targets: HRMTargets(fraction: 1.0, ptt: 0.0, stop: 0.0),
            carry: prediction.nextCarry
        )

        #expect(prediction.nextCarry.zH.count == 4)
        #expect(prediction.nextCarry.zL.count == 4)
        #expect(training.nextCarry.zH.count == 4)
        #expect(training.nextCarry.zL.count == 4)
        #expect(prediction.trace.hPasses == 2)
        #expect(prediction.trace.lPasses == 6)
        #expect(training.trace.cycleOrder.last == "H1")
    }

    @Test("Task heads read high-level state")
    func testTaskHeadsReadHighLevelState() {
        var model = HRMModel(hDim: 16, zDim: 4, xPixels: 4)
        model.registerEdges(["ETH-USD"])
        let snapshot = model.predictorSnapshot(edge: "ETH-USD")

        #expect(snapshot != nil)
        #expect(snapshot?.fractionHead.count == model.hDim)
        #expect(snapshot?.pttHead.count == model.hDim)
        #expect(snapshot?.stopHead.count == model.hDim)
    }

    @Test("Top-left anchor preserved on square growth")
    func testTopLeftAnchorPreservedOnSquareGrowth() {
        var model = HRMModel(hDim: 4, zDim: 4, xPixels: 4)
        model.registerEdges(["SOL-USD"])
        let before = model.predictorSnapshot(edge: "SOL-USD")!
        let oldAnchor = before.hToH

        model.grow(dim: .h, geometryPolicy: nil)

        let after = model.predictorSnapshot(edge: "SOL-USD")!
        #expect(model.hDim == 16)
        #expect(topLeftBlock(after.hToH, size: oldAnchor.count) == oldAnchor)
        #expect(model.growthMetadata.activePowers == [4, 16])
    }

    @Test("1D fallback uses 0-180-180-0")
    func testOneDimensionalFallback() {
        var model = HRMModel(hDim: 4, zDim: 4, xPixels: 4)
        model.registerEdges(["DOGE-USD"])
        let before = model.predictorSnapshot(edge: "DOGE-USD")!

        model.grow(dim: .h, geometryPolicy: nil)

        let after = model.predictorSnapshot(edge: "DOGE-USD")!
        let blockWidth = before.hToL[0].count
        let original = before.hToL
        let rotated = matrixColumnBlock(after.hToL, block: 1, blockWidth: blockWidth)
        let rotated2 = matrixColumnBlock(after.hToL, block: 2, blockWidth: blockWidth)
        let finalBlock = matrixColumnBlock(after.hToL, block: 3, blockWidth: blockWidth)

        #expect(matrixColumnBlock(after.hToL, block: 0, blockWidth: blockWidth) == original)
        #expect(rotated == rotate180ForTest(original))
        #expect(rotated2 == rotate180ForTest(original))
        #expect(finalBlock == original)
    }

    @Test("Bridge is not default growth policy")
    func testBridgeIsNotDefault() {
        var model = HRMModel(hDim: 4, zDim: 4, xPixels: 4)
        model.registerEdges(["XRP-USD"])
        model.grow(dim: .h, geometryPolicy: nil)

        #expect(model.growthMetadata.bridgeDefaultEnabled == false)
        let latestPolicies = model.growthMetadata.history.last.map { Array($0.tensorPolicies.values) } ?? []
        #expect(!latestPolicies.contains(GrowthGeometryPolicy.bridgeCompatibility))
    }

    @Test("Checkpoint save and load restores growth metadata")
    func testCheckpointRoundTrip() throws {
        let checkpoint = temporaryFilePath(named: "hrm_growth_roundtrip.json")

        var model = HRMModel(hDim: 4, zDim: 4, xPixels: 4)
        model.registerEdges(["BTC-USD"])
        _ = model.predict(edge: "BTC-USD", fisheye: [0.1, 0.2, 0.3, 0.4], carry: nil)
        model.grow(dim: .z, geometryPolicy: nil)
        try model.save(path: checkpoint)

        var restored = HRMModel(hDim: 4, zDim: 4, xPixels: 4)
        restored.registerEdges(["BTC-USD"])
        try restored.load(path: checkpoint)

        #expect(restored.hDim == model.hDim)
        #expect(restored.zDim == model.zDim)
        #expect(restored.growthMetadata == model.growthMetadata)
        #expect(restored.predictorSnapshot(edge: "BTC-USD") == model.predictorSnapshot(edge: "BTC-USD"))
    }

    @Test("Fisheye boundaries calculation")
    func testFisheyeBoundaries() {
        let boundaries = fisheyeBoundaries(yDepth: 200, xPixels: 20, curvature: 2.0)

        #expect(boundaries.count == 20)
        #expect(boundaries[0] >= 0)
        #expect(boundaries.last! <= 200)
        for index in 1..<boundaries.count {
            #expect(boundaries[index] >= boundaries[index - 1])
        }
    }

    @Test("Fisheye sample calculation")
    func testFisheyeSample() {
        let candles: [Double] = [100.0, 101.0, 102.0, 103.0, 104.0]
        let boundaries = [1, 2, 3, 4]
        let samples = fisheyeSample(candles: candles, boundaries: boundaries, xPixels: 4)

        #expect(samples.count == 4)
        #expect(samples.allSatisfy { $0.isFinite })
    }

    @Test("Matrix operations")
    func testMatrixOperations() {
        let result = matvec(A: [[1.0, 2.0], [3.0, 4.0]], x: [1.0, 1.0])
        #expect(result.count == 2)
        #expect(result[0] == 3.0)
        #expect(result[1] == 7.0)
    }

    @Test("Dot product")
    func testDotProduct() {
        #expect(dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0)
    }

    @Test("Sigmoid function")
    func testSigmoid() {
        #expect(sigmoid(0.0) == 0.5)
        #expect(abs(sigmoid(1.0) - 0.7310585786300049) < 1e-10)
        #expect(abs(sigmoid(-1.0) - 0.2689414213699951) < 1e-10)
    }

    @Test("BCE loss")
    func testBCELoss() {
        let loss = bceLoss(pred: 0.8, target: 1.0)
        let betterLoss = bceLoss(pred: 0.9, target: 1.0)

        #expect(loss > 0)
        #expect(loss.isFinite)
        #expect(betterLoss < loss)
    }
}

private func matrixColumnBlock(_ matrix: [[Double]], block: Int, blockWidth: Int) -> [[Double]] {
    matrix.map { row in
        let start = block * blockWidth
        let end = start + blockWidth
        return Array(row[start..<end])
    }
}

private func rotate180ForTest(_ matrix: [[Double]]) -> [[Double]] {
    matrix.reversed().map { Array($0.reversed()) }
}

private func temporaryFilePath(named name: String) -> String {
    FileManager.default.temporaryDirectory.appendingPathComponent(name).path
}
