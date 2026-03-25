//
//  HRMModelTests.swift
//  AutotradeTests
//
//  Tests for HRM model functionality
//

import Testing
import AutotradeHRM

@Suite("HRM Model Tests")
struct HRMModelTests {
    @Test("HRMModel initialization")
    func testHRMModelInitialization() {
        var model = HRMModel(
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
    }

    @Test("Square invariant enforcement")
    func testSquareInvariant() {
        // Should work with square dimensions
        let model = HRMModel(hDim: 8, zDim: 8)
        #expect(model.hDim == 8)
        #expect(model.zDim == 8)

        // Should enforce square even if not provided
        let model2 = HRMModel(hDim: 16, zDim: 8)
        #expect(model2.hDim == 8) // Should be corrected to match zDim
        #expect(model2.zDim == 8)
    }

    @Test("Edge registration")
    func testEdgeRegistration() {
        var model = HRMModel(nEdges: 0)
        let testEdges = ["BTC-USD", "ETH-USD", "SOL-USD"]

        model.registerEdges(testEdges)

        #expect(model.edgeNames == testEdges)
        #expect(model.edgeNames.count == 3)
    }

    @Test("Fisheye boundaries calculation")
    func testFisheyeBoundaries() {
        let yDepth = 200
        let xPixels = 20
        let curvature = 2.0

        let boundaries = fisheyeBoundaries(yDepth: yDepth, xPixels: xPixels, curvature: curvature)

        #expect(boundaries.count == xPixels)
        #expect(boundaries[0] >= 0)
        #expect(boundaries.last! <= yDepth)

        // Should be monotonically increasing
        for i in 1..<boundaries.count {
            #expect(boundaries[i] >= boundaries[i-1])
        }
    }

    @Test("Fisheye sample calculation")
    func testFisheyeSample() {
        let candles: [Double] = [100.0, 101.0, 102.0, 103.0, 104.0]
        let boundaries = [1, 2, 3, 4]
        let xPixels = 4

        let samples = fisheyeSample(candles: candles, boundaries: boundaries, xPixels: xPixels)

        #expect(samples.count == xPixels)
        #expect(samples.allSatisfy { $0.isFinite })
    }

    @Test("Matrix operations")
    func testMatrixOperations() {
        let A = [[1.0, 2.0], [3.0, 4.0]]
        let x = [1.0, 1.0]

        let result = matvec(A: A, x: x)
        #expect(result.count == 2)
        #expect(result[0] == 3.0) // 1*1 + 2*1
        #expect(result[1] == 7.0) // 3*1 + 4*1
    }

    @Test("Dot product")
    func testDotProduct() {
        let a = [1.0, 2.0, 3.0]
        let b = [4.0, 5.0, 6.0]

        let result = dot(a, b)
        #expect(result == 32.0) // 1*4 + 2*5 + 3*6
    }

    @Test("Sigmoid function")
    func testSigmoid() {
        #expect(sigmoid(0.0) == 0.5)
        #expect(abs(sigmoid(1.0) - 0.7310585786300049) < 1e-10)
        #expect(abs(sigmoid(-1.0) - 0.2689414213699951) < 1e-10)
    }

    @Test("BCE loss")
    func testBCELoss() {
        let pred = 0.8
        let target = 1.0
        let loss = bceLoss(pred: pred, target: target)

        #expect(loss > 0)
        #expect(loss.isFinite)

        // Loss should be lower when prediction matches target
        let betterPred = 0.9
        let betterLoss = bceLoss(pred: betterPred, target: target)
        #expect(betterLoss < loss)
    }
}