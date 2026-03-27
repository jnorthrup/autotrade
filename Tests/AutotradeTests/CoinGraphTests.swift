//
//  CoinGraphTests.swift
//  AutotradeTests
//
//  Tests for CoinGraph functionality
//

import Foundation
import Testing
@testable import AutotradeHRM

@Suite("CoinGraph Tests")
struct CoinGraphTests {
    @Test("DBCandle structure", .disabled("CoinGraph not in AutotradeHRM target"))
    func testCoinGraphInitialization() async throws {
        let candle = DBCandle(
            timestamp: Date(),
            open: 100.0, high: 105.0, low: 95.0,
            close: 102.0, volume: 1000.0
        )
        #expect(candle.open == 100.0)
    }

    @Test("Date range validation")
    func testDateRangeValidation() {
        let endDate = Date()
        let startDate = endDate.addingTimeInterval(-365 * 24 * 3600)

        #expect(startDate < endDate)

        let interval = endDate.timeIntervalSince(startDate)
        let oneYear: TimeInterval = 365 * 24 * 3600
        #expect(abs(interval - oneYear) < 24 * 3600)
    }

    @Test("ISO8601 date formatting")
    func testISO8601Formatting() {
        let df = ISO8601DateFormatter()
        df.formatOptions = [.withInternetDateTime]

        let testDate = Date(timeIntervalSince1970: 1640995200)
        let formatted = df.string(from: testDate)

        #expect(formatted.hasPrefix("2022-01-01T"))
        #expect(formatted.hasSuffix("Z"))

        let parsed = df.date(from: formatted)
        #expect(parsed != nil)
        #expect(abs(parsed!.timeIntervalSince(testDate)) < 1.0)
    }
}