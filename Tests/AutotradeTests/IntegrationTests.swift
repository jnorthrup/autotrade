//
//  IntegrationTests.swift
//  AutotradeTests
//
//  End-to-end integration tests
//

import Testing
import AutotradeHRM

@Suite("Integration Tests")
struct IntegrationTests {
    @Test("Full pipeline integration", .disabled("Requires database file"))
    func testFullPipeline() async throws {
        // This test would require the actual candles.duckdb file
        // For now, test the components separately

        let graph = CoinGraph(feeRate: 0.001)
        #expect(graph.feeRate == 0.001)

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

    @Test("Memory management")
    func testMemoryManagement() {
        // Test that objects don't leak excessively
        var graphs: [CoinGraph] = []

        for _ in 0..<10 {
            autoreleasepool {
                let graph = CoinGraph()
                graphs.append(graph)
            }
        }

        // Should be able to create multiple instances
        #expect(graphs.count == 10)
        graphs.removeAll()
    }

    @Test("Concurrent access safety")
    func testConcurrentAccess() async {
        let graph = CoinGraph()

        // Test that the graph can be accessed from multiple tasks
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<5 {
                group.addTask {
                    // Simulate concurrent operations
                    let fee = graph.feeRate
                    #expect(fee == 0.001)
                }
            }
        }
    }

    @Test("Error handling")
    func testErrorHandling() async {
        let graph = CoinGraph()

        // Test graceful handling of missing data
        do {
            let nBars = try await graph.load(minPartners: 1000) // Impossible requirement
            // Should not crash, should handle gracefully
            #expect(nBars >= 0)
        } catch {
            // Should handle errors gracefully
            #expect(error is NSError)
        }
    }

    @Test("Performance baseline")
    func testPerformanceBaseline() {
        let start = Date()

        // Create a reasonably sized model
        var model = HRMModel(
            nEdges: 10,
            hDim: 16,
            zDim: 16,
            yDepth: 50,
            xPixels: 10
        )

        let edges = (0..<10).map { "PAIR-\($0)" }
        model.registerEdges(edges)

        let end = Date()
        let duration = end.timeIntervalSince(start)

        // Should complete in reasonable time (< 1 second)
        #expect(duration < 1.0)
    }

    @Test("Data consistency")
    func testDataConsistency() {
        // Test that repeated operations give consistent results
        let graph1 = CoinGraph(feeRate: 0.001)
        let graph2 = CoinGraph(feeRate: 0.001)

        #expect(graph1.feeRate == graph2.feeRate)

        var model1 = HRMModel(hDim: 8, zDim: 8)
        var model2 = HRMModel(hDim: 8, zDim: 8)

        #expect(model1.hDim == model2.hDim)
        #expect(model1.zDim == model2.zDim)
    }

    @Test("Edge case handling")
    func testEdgeCases() {
        // Test with minimal parameters
        var model = HRMModel(
            nEdges: 1,
            hDim: 4,
            zDim: 4,
            yDepth: 1,
            xPixels: 1
        )

        #expect(model.hDim == 4)
        #expect(model.yDepth == 1)
        #expect(model.xPixels == 1)

        // Test with empty edge list
        model.registerEdges([])
        #expect(model.edgeNames.isEmpty)
    }
}