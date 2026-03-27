//
//  MockDataTests.swift
//  AutotradeTests
//
//  Tests using mock data for components that require database
//

import Foundation
import Testing
@testable import AutotradeHRM

@Suite("Mock Data Tests")
struct MockDataTests {
    @Test("Mock candle data validation")
    func testMockCandleData() {
        let mockCandles = createMockCandles(count: 100)

        #expect(mockCandles.count == 100)

        // Validate OHLC relationships
        for candle in mockCandles {
            #expect(candle.high >= candle.low)
            #expect(candle.high >= max(candle.open, candle.close))
            #expect(candle.low <= min(candle.open, candle.close))
            #expect(candle.volume >= 0)
            #expect(candle.open > 0)
            #expect(candle.close > 0)
        }

        // Check temporal ordering
        for i in 1..<mockCandles.count {
            #expect(mockCandles[i].timestamp >= mockCandles[i-1].timestamp)
        }
    }

    @Test("Mock graph construction")
    func testMockGraphConstruction() {
        let mockCandles = createMockCandles(count: 50)
        let mockPairs = ["BTC-USD", "ETH-USD", "SOL-USD"]

        // Simulate graph construction
        var edges: [String: [DBCandle]] = [:]
        for pair in mockPairs {
            edges[pair] = mockCandles
        }

        #expect(edges.count == 3)
        #expect(edges["BTC-USD"]?.count == 50)
    }

    @Test("Velocity calculation with mock data")
    func testVelocityCalculation() {
        let candles = createMockCandles(count: 10)

        // Calculate velocities
        var velocities: [Double] = []
        for i in 1..<candles.count {
            let prev = candles[i-1]
            let curr = candles[i]
            if prev.close > 0 {
                let velocity = log(curr.close / prev.close)
                velocities.append(velocity)
                #expect(velocity.isFinite)
            }
        }

        #expect(!velocities.isEmpty)
        #expect(velocities.count == 9) // One less than candle count
    }

    @Test("Fisheye processing with mock data")
    func testFisheyeProcessing() {
        let candles = createMockCandles(count: 200)
        let closes = candles.map { $0.close }

        let boundaries = fisheyeBoundaries(yDepth: 200, xPixels: 20, curvature: 2.0)
        let fisheyeData = fisheyeSample(candles: closes, boundaries: boundaries, xPixels: 20)

        #expect(fisheyeData.count == 20)
        #expect(fisheyeData.allSatisfy { $0.isFinite })
    }

    @Test("Model prediction with mock data")
    func testModelPredictionWithMockData() {
        let mockCandles = createMockCandles(count: 100)

        let xPixels = 20
        var model = HRMModel(
            nEdges: 1,
            hDim: 4,
            zDim: 4,
            xPixels: xPixels
        )

        model.registerEdges(["MOCK-USD"])

        // Feed enough bars so fisheye buffer fills up
        let timestamps = mockCandles.map { $0.timestamp }
        let edges: [String: [DBCandle]] = ["MOCK-USD": mockCandles]

        // Advance through bars to populate closeBuffer
        var predictions: [String: (fraction: Double, ptt: Double, stop: Double)] = [:]
        for i in 0..<min(100, timestamps.count) {
            predictions = model.predict(commonTimestamps: timestamps, edges: edges, barIdx: i)
        }

        #expect(predictions["MOCK-USD"] != nil)
        if let pred = predictions["MOCK-USD"] {
            #expect(pred.fraction.isFinite)
            #expect(pred.ptt.isFinite)
            #expect(pred.stop.isFinite)
        }
    }

    @Test("Batch processing simulation")
    func testBatchProcessingSimulation() {
        let batchSize = 32
        let sequenceLength = 100

        // Create mock batch data
        var batchData: [[Double]] = []
        for _ in 0..<batchSize {
            let sequence = (0..<sequenceLength).map { _ in Double.random(in: 0.95...1.05) }
            batchData.append(sequence)
        }

        #expect(batchData.count == batchSize)
        #expect(batchData.allSatisfy { $0.count == sequenceLength })

        // Simulate processing
        let processed = batchData.map { sequence in
            sequence.map { $0 * 1.001 } // Simple transformation
        }

        #expect(processed.count == batchSize)
        #expect(processed.allSatisfy { $0.count == sequenceLength })
        #expect(processed.allSatisfy { sequence in
            zip(sequence, batchData[processed.firstIndex(of: sequence)!]).allSatisfy { $0.0 > $0.1 }
        })
    }
}

// Mock data generation
private func createMockCandles(count: Int) -> [DBCandle] {
    var candles: [DBCandle] = []
    let basePrice = 50000.0
    let volatility = 0.02
    let interval: TimeInterval = 3600 // 1 hour

    var currentPrice = basePrice
    var currentTime = Date()

    for _ in 0..<count {
        let priceChange = Double.random(in: -volatility...volatility)
        currentPrice *= (1.0 + priceChange)

        let open = currentPrice
        let close = currentPrice * (1.0 + Double.random(in: -volatility/2...volatility/2))
        let high = max(open, close) * (1.0 + abs(Double.random(in: 0...volatility/4)))
        let low = min(open, close) * (1.0 - abs(Double.random(in: 0...volatility/4)))
        let volume = Double.random(in: 100...10000)

        let candle = DBCandle(
            timestamp: currentTime,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )

        candles.append(candle)
        currentTime = currentTime.addingTimeInterval(interval)
    }

    return candles
}
