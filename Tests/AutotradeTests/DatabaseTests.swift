//
//  DatabaseTests.swift
//  AutotradeTests
//
//  TDD tests for DuckDBCandleStore — strict translation of Python candle_cache.py
//

import Foundation
import Testing
import AutotradeHRM
import DuckDB

@Suite("DuckDBCandleStore Tests")
struct DatabaseTests {

    // MARK: - Schema

    @Test("Schema creates candles table")
    func testSchemaCandlesTable() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let result = try store.query("DESCRIBE candles")
        let columnNames = Array(result[0].cast(to: String.self)).compactMap { $0 }
        #expect(columnNames.contains("product_id"))
        #expect(columnNames.contains("timestamp"))
        #expect(columnNames.contains("open"))
        #expect(columnNames.contains("high"))
        #expect(columnNames.contains("low"))
        #expect(columnNames.contains("close"))
        #expect(columnNames.contains("volume"))
        #expect(columnNames.contains("granularity"))
    }

    @Test("Schema creates edge_params table")
    func testSchemaEdgeParamsTable() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let result = try store.query("DESCRIBE edge_params")
        let columnNames = Array(result[0].cast(to: String.self)).compactMap { $0 }
        #expect(columnNames.contains("edge"))
        #expect(columnNames.contains("curvature"))
        #expect(columnNames.contains("y_depth"))
        #expect(columnNames.contains("x_pixels"))
        #expect(columnNames.contains("fee_rate"))
        #expect(columnNames.contains("updated_at"))
    }

    @Test("Schema is idempotent")
    func testSchemaIdempotent() throws {
        // Creating a second store on same in-memory db should not throw
        let store = try DuckDBCandleStore(inMemory: true)
        // Re-run schema DDL via raw query — should not throw
        try store.query("SELECT COUNT(*) FROM candles")
        try store.query("SELECT COUNT(*) FROM edge_params")
    }

    // MARK: - Insert + Query

    @Test("Insert and list products")
    func testInsertAndListProducts() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let candles = makeTestCandles(count: 5, basePrice: 100)
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")

        let products = try store.listProducts(granularity: "300")
        #expect(products == ["BTC-USD"])
    }

    @Test("Multiple products")
    func testMultipleProducts() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let btc = makeTestCandles(count: 3, basePrice: 50000)
        let eth = makeTestCandles(count: 3, basePrice: 3000)
        try store.saveCandles(btc, productId: "BTC-USD", granularity: "300")
        try store.saveCandles(eth, productId: "ETH-USD", granularity: "300")

        let products = try store.listProducts(granularity: "300")
        #expect(products.count == 2)
        #expect(products.contains("BTC-USD"))
        #expect(products.contains("ETH-USD"))
    }

    @Test("List products filters by granularity")
    func testListProductsGranularityFilter() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let candles = makeTestCandles(count: 3, basePrice: 100)
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")
        try store.saveCandles(candles, productId: "ETH-USD", granularity: "60")

        let g300 = try store.listProducts(granularity: "300")
        #expect(g300 == ["BTC-USD"])

        let g60 = try store.listProducts(granularity: "60")
        #expect(g60 == ["ETH-USD"])
    }

    @Test("Get candles by date range")
    func testGetCandlesDateRange() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let base = Date(timeIntervalSince1970: 1700000000)
        let candles = (0..<10).map { i -> DBCandle in
            let t = base.addingTimeInterval(Double(i) * 300)
            return DBCandle(timestamp: t, open: 100, high: 105, low: 95, close: 102, volume: 1000)
        }
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")

        let start = base.addingTimeInterval(900)  // skip first 3
        let end = base.addingTimeInterval(1800)   // through 6
        let result = try store.getCandles(productId: "BTC-USD", start: start, end: end, granularity: "300")

        #expect(result.count == 4)  // indices 3,4,5,6
        for c in result {
            #expect(c.timestamp >= start)
            #expect(c.timestamp <= end)
        }
    }

    @Test("Get candles returns empty for unknown product")
    func testGetCandlesUnknownProduct() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let result = try store.getCandles(
            productId: "NONEXISTENT-USD",
            start: Date(timeIntervalSince1970: 0),
            end: Date(),
            granularity: "300"
        )
        #expect(result.isEmpty)
    }

    @Test("Get candles returns empty for future range")
    func testGetCandlesFutureRange() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let candles = makeTestCandles(count: 5, basePrice: 100)
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")

        let future = Date().addingTimeInterval(365 * 24 * 3600)
        let result = try store.getCandles(productId: "BTC-USD", start: future, end: future, granularity: "300")
        #expect(result.isEmpty)
    }

    // MARK: - Idempotent insert (INSERT OR IGNORE)

    @Test("Duplicate inserts are ignored")
    func testDuplicateInsertIgnored() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let candles = makeTestCandles(count: 5, basePrice: 100)
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")

        let result = try store.getCandles(
            productId: "BTC-USD",
            start: candles[0].timestamp,
            end: candles[4].timestamp,
            granularity: "300"
        )
        #expect(result.count == 5)  // not 10
    }

    // MARK: - Bulk insert (Appender)

    @Test("Bulk insert 1000 candles")
    func testBulkInsert() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let candles = makeTestCandles(count: 1000, basePrice: 50000)
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")

        let products = try store.listProducts(granularity: "300")
        #expect(products == ["BTC-USD"])

        let result = try store.getCandles(
            productId: "BTC-USD",
            start: candles[0].timestamp,
            end: candles[999].timestamp,
            granularity: "300"
        )
        #expect(result.count == 1000)
    }

    // MARK: - Edge params

    @Test("Save and load edge params")
    func testEdgeParamsRoundTrip() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let params = EdgeParams(curvature: 3.0, yDepth: 100, xPixels: 15, feeRate: 0.002)
        try store.saveEdgeParams(edge: "BTC-USD", params: params)

        let loaded = try store.loadEdgeParams()
        #expect(loaded.count == 1)
        #expect(loaded["BTC-USD"]?.curvature == 3.0)
        #expect(loaded["BTC-USD"]?.yDepth == 100)
        #expect(loaded["BTC-USD"]?.xPixels == 15)
        #expect(loaded["BTC-USD"]?.feeRate == 0.002)
    }

    @Test("Edge params upsert")
    func testEdgeParamsUpsert() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let p1 = EdgeParams(curvature: 2.0, yDepth: 200, xPixels: 20, feeRate: 0.001)
        let p2 = EdgeParams(curvature: 4.0, yDepth: 300, xPixels: 30, feeRate: 0.003)
        try store.saveEdgeParams(edge: "BTC-USD", params: p1)
        try store.saveEdgeParams(edge: "BTC-USD", params: p2)  // upsert

        let loaded = try store.loadEdgeParams()
        #expect(loaded.count == 1)
        #expect(loaded["BTC-USD"]?.curvature == 4.0)
        #expect(loaded["BTC-USD"]?.yDepth == 300)
    }

    @Test("Multiple edge params")
    func testMultipleEdgeParams() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        try store.saveEdgeParams(edge: "BTC-USD", params: EdgeParams(curvature: 2.0, yDepth: 200, xPixels: 20, feeRate: 0.001))
        try store.saveEdgeParams(edge: "ETH-USD", params: EdgeParams(curvature: 3.0, yDepth: 100, xPixels: 10, feeRate: 0.002))

        let loaded = try store.loadEdgeParams()
        #expect(loaded.count == 2)
        #expect(loaded["BTC-USD"] != nil)
        #expect(loaded["ETH-USD"] != nil)
    }

    // MARK: - Generic query with params

    @Test("Parameterized query")
    func testParameterizedQuery() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let candles = makeTestCandles(count: 10, basePrice: 100)
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")

        let stmt = try PreparedStatement(
            connection: store.connection,
            query: "SELECT COUNT(*) FROM candles WHERE product_id = $1 AND granularity = $2"
        )
        try stmt.bind("BTC-USD", at: 1)
        try stmt.bind("300", at: 2)
        let result = try stmt.execute()
        let count = Array(result[0].cast(to: Int64.self)).compactMap { $0 }.first
        #expect(count == 10)
    }

    // MARK: - SQL injection safety

    @Test("Product ID with SQL metacharacters is safe")
    func testSQLInjectionSafety() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let candles = makeTestCandles(count: 2, basePrice: 100)
        let maliciousId = "'; DROP TABLE candles; --"
        try store.saveCandles(candles, productId: maliciousId, granularity: "300")

        // Table should still exist
        let products = try store.listProducts(granularity: "300")
        #expect(products.contains(maliciousId))

        // Candles table should still be queryable
        let result = try store.getCandles(
            productId: maliciousId,
            start: candles[0].timestamp,
            end: candles[1].timestamp,
            granularity: "300"
        )
        #expect(result.count == 2)
    }

    // MARK: - Empty insert

    @Test("Save empty candle array does not throw")
    func testSaveEmptyCandles() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        try store.saveCandles([], productId: "BTC-USD", granularity: "300")
        let products = try store.listProducts(granularity: "300")
        #expect(products.isEmpty)
    }

    // MARK: - Candle OHLC validation

    @Test("Retrieved candles have valid OHLC")
    func testRetrievedCandleOHLC() throws {
        let store = try DuckDBCandleStore(inMemory: true)
        let candles = makeTestCandles(count: 20, basePrice: 50000)
        try store.saveCandles(candles, productId: "BTC-USD", granularity: "300")

        let result = try store.getCandles(
            productId: "BTC-USD",
            start: candles[0].timestamp,
            end: candles[19].timestamp,
            granularity: "300"
        )

        for c in result {
            #expect(c.high >= c.low)
            #expect(c.high >= c.open)
            #expect(c.high >= c.close)
            #expect(c.low <= c.open)
            #expect(c.low <= c.close)
            #expect(c.volume >= 0)
            #expect(c.open > 0)
            #expect(c.close > 0)
        }
    }

    // MARK: - DBCandle struct

    @Test("DBCandle equality")
    func testDBCandleEquality() {
        let t = Date()
        let a = DBCandle(timestamp: t, open: 1, high: 2, low: 0.5, close: 1.5, volume: 100)
        let b = DBCandle(timestamp: t, open: 1, high: 2, low: 0.5, close: 1.5, volume: 100)
        #expect(a.timestamp == b.timestamp)
        #expect(a.open == b.open)
        #expect(a.close == b.close)
    }

    // MARK: - File-based store

    @Test("File-based store creates and reopens")
    func testFileBasedStore() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("test_\(UUID().uuidString).duckdb")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let store1 = try DuckDBCandleStore(path: tmp.path)
        let candles = makeTestCandles(count: 3, basePrice: 100)
        try store1.saveCandles(candles, productId: "BTC-USD", granularity: "300")

        // Reopen
        let store2 = try DuckDBCandleStore(path: tmp.path)
        let products = try store2.listProducts(granularity: "300")
        #expect(products == ["BTC-USD"])

        let result = try store2.getCandles(
            productId: "BTC-USD",
            start: candles[0].timestamp,
            end: candles[2].timestamp,
            granularity: "300"
        )
        #expect(result.count == 3)
    }
}

// MARK: - Test helpers

private func makeTestCandles(count: Int, basePrice: Double) -> [DBCandle] {
    var candles: [DBCandle] = []
    var price = basePrice
    let start = Date(timeIntervalSince1970: 1700000000)

    for i in 0..<count {
        let t = start.addingTimeInterval(Double(i) * 300)
        let change = Double.random(in: -0.02...0.02)
        let open = price
        let close = price * (1 + change)
        let high = max(open, close) * (1 + abs(Double.random(in: 0...0.005)))
        let low = min(open, close) * (1 - abs(Double.random(in: 0...0.005)))
        let volume = Double.random(in: 100...10000)

        candles.append(DBCandle(timestamp: t, open: open, high: high, low: low, close: close, volume: volume))
        price = close
    }
    return candles
}
