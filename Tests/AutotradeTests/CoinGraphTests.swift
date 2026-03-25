//
//  CoinGraphTests.swift
//  AutotradeTests
//
//  Tests for CoinGraph functionality
//

import Testing
import AutotradeHRM

@Suite("CoinGraph Tests")
struct CoinGraphTests {
    @Test("CoinGraph initialization")
    func testCoinGraphInitialization() async throws {
        let graph = CoinGraph(feeRate: 0.001)
        #expect(graph.feeRate == 0.001)
        #expect(graph.nodes.isEmpty)
        #expect(graph.allPairs.isEmpty)
        #expect(graph.edges.isEmpty)
    }

    @Test("Date range validation")
    func testDateRangeValidation() {
        let endDate = Date()
        let startDate = endDate.addingTimeInterval(-365 * 24 * 3600)

        // Start should be before end
        #expect(startDate < endDate)

        // Should be approximately 1 year apart
        let interval = endDate.timeIntervalSince(startDate)
        let oneYear: TimeInterval = 365 * 24 * 3600
        #expect(abs(interval - oneYear) < 24 * 3600) // Within 1 day
    }

    @Test("ISO8601 date formatting")
    func testISO8601Formatting() {
        let df = ISO8601DateFormatter()
        df.formatOptions = [.withInternetDateTime]

        let testDate = Date(timeIntervalSince1970: 1640995200) // 2022-01-01 00:00:00 UTC
        let formatted = df.string(from: testDate)

        // Should be in ISO8601 format
        #expect(formatted.hasPrefix("2022-01-01T"))
        #expect(formatted.hasSuffix("Z"))

        // Should be parseable back
        let parsed = df.date(from: formatted)
        #expect(parsed != nil)
        #expect(abs(parsed!.timeIntervalSince(testDate)) < 1.0)
    }
}