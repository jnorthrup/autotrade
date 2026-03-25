//
//  DatabaseTests.swift
//  AutotradeTests
//
//  Tests for DuckDB integration
//

import Testing
import AutotradeHRM

@Suite("Database Tests")
struct DatabaseTests {
    @Test("Database connection setup")
    func testDatabaseConnection() throws {
        let db = DuckDBFFI(path: "candles.duckdb")

        // Should be able to query products without throwing
        let products = try db.queryProducts()
        #expect(products is [String])
    }

    @Test("Product query returns reasonable data")
    func testProductQuery() throws {
        let db = DuckDBFFI()

        let products = try db.queryProducts()

        // Should have some products
        #expect(!products.isEmpty)

        // Should contain common pairs
        let productStrings = products.joined(separator: " ")
        #expect(productStrings.contains("BTC") || productStrings.contains("USD"))
    }

    @Test("Candle query validation")
    func testCandleQuery() throws {
        let db = DuckDBFFI()
        let endDate = Date()
        let startDate = endDate.addingTimeInterval(-7 * 24 * 3600) // 1 week

        let candles = try db.queryCandles(product: "BTC-USD", start: startDate, end: endDate)

        // Should return some candles or empty array (if no data in range)
        #expect(candles is [DBCandle])

        // If candles exist, they should be in date range
        for candle in candles {
            #expect(candle.timestamp >= startDate)
            #expect(candle.timestamp <= endDate)
            #expect(candle.open > 0)
            #expect(candle.high >= candle.open)
            #expect(candle.low <= candle.open)
            #expect(candle.close > 0)
            #expect(candle.volume >= 0)
        }
    }

    @Test("DBCandle structure validation")
    func testDBCandleStructure() {
        let testDate = Date()
        let candle = DBCandle(
            timestamp: testDate,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000.0
        )

        #expect(candle.timestamp == testDate)
        #expect(candle.open == 100.0)
        #expect(candle.high == 105.0)
        #expect(candle.low == 95.0)
        #expect(candle.close == 102.0)
        #expect(candle.volume == 1000.0)
    }

    @Test("Date range edge cases")
    func testDateRangeEdgeCases() throws {
        let db = DuckDBFFI()

        // Future date range should return empty
        let futureDate = Date().addingTimeInterval(365 * 24 * 3600)
        let candles = try db.queryCandles(product: "BTC-USD", start: futureDate, end: futureDate)
        #expect(candles.isEmpty)

        // Very old date range should return empty
        let oldDate = Date(timeIntervalSince1970: 0)
        let ancientCandles = try db.queryCandles(product: "BTC-USD", start: oldDate, end: oldDate.addingTimeInterval(3600))
        #expect(ancientCandles.isEmpty)
    }

    @Test("Invalid product handling")
    func testInvalidProduct() throws {
        let db = DuckDBFFI()

        // Query for non-existent product
        let candles = try db.queryCandles(product: "NONEXISTENT-USD", start: Date().addingTimeInterval(-3600), end: Date())
        #expect(candles.isEmpty)
    }
}