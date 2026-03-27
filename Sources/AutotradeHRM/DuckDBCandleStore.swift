//
//  DuckDBCandleStore.swift
//  AutotradeHRM
//
//  Strict Swift translation of Python candle_cache.py.
//  Uses DuckDB Swift FFI with prepared statements (no CLI shell-out).
//

import Foundation
import DuckDB

// MARK: - Edge params (matches Python candle_cache.py edge_params table)

public struct EdgeParams: Sendable, Codable {
    public var curvature: Double
    public var yDepth: Int
    public var xPixels: Int
    public var feeRate: Double

    public init(curvature: Double, yDepth: Int, xPixels: Int, feeRate: Double) {
        self.curvature = curvature
        self.yDepth = yDepth
        self.xPixels = xPixels
        self.feeRate = feeRate
    }
}

// MARK: - Store protocol

public protocol CandleStore {
    func listProducts(granularity: String) throws -> [String]
    func getCandles(productId: String, start: Foundation.Date, end: Foundation.Date, granularity: String) throws -> [DBCandle]
    func saveCandles(_ candles: [DBCandle], productId: String, granularity: String) throws
    func saveEdgeParams(edge: String, params: EdgeParams) throws
    func loadEdgeParams() throws -> [String: EdgeParams]
}

// MARK: - DuckDB implementation

public final class DuckDBCandleStore: CandleStore, @unchecked Sendable {
    public let database: Database
    public let connection: Connection
    private let queue = DispatchQueue(label: "com.autotrade.duckdb.store")

    /// Open (or create) a file-based store
    public init(path: String) throws {
        database = try Database(store: .file(at: URL(fileURLWithPath: path)))
        connection = try database.connect()
        try Self.createSchema(on: connection)
    }

    /// Create an in-memory store for testing
    public init(inMemory: Bool) throws {
        guard inMemory else { fatalError("Use init(path:) for file stores") }
        database = try Database(store: .inMemory)
        connection = try database.connect()
        try Self.createSchema(on: connection)
    }

    // MARK: - Schema (matches Python _init_db)

    static func createSchema(on conn: Connection) throws {
        try conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                product_id VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                granularity VARCHAR,
                PRIMARY KEY (product_id, timestamp, granularity)
            )
        """)
        try conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(timestamp)")
        try conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_prod_gran ON candles(product_id, granularity)")

        try conn.execute("""
            CREATE TABLE IF NOT EXISTS edge_params (
                edge VARCHAR PRIMARY KEY,
                curvature DOUBLE DEFAULT 2.0,
                y_depth INTEGER DEFAULT 200,
                x_pixels INTEGER DEFAULT 20,
                fee_rate DOUBLE DEFAULT 0.001,
                updated_at TIMESTAMP DEFAULT now()
            )
        """)
    }

    // MARK: - Generic query (matches Python CandleCache.query)

    public func query(_ sql: String) throws -> ResultSet {
        try queue.sync { try connection.query(sql) }
    }

    // MARK: - List products (matches Python CandleCache.list_products)

    public func listProducts(granularity: String = "300") throws -> [String] {
        try queue.sync {
            let stmt = try PreparedStatement(
                connection: connection,
                query: "SELECT DISTINCT product_id FROM candles WHERE granularity = $1"
            )
            try stmt.bind(granularity, at: 1)
            let result = try stmt.execute()
            return Array(result[0].cast(to: String.self)).compactMap { $0 }
        }
    }

    // MARK: - Get candles (matches Python CandleCache.get_candles)

    public func getCandles(
        productId: String,
        start: Foundation.Date,
        end: Foundation.Date,
        granularity: String = "300"
    ) throws -> [DBCandle] {
        try queue.sync {
            let stmt = try PreparedStatement(
                connection: connection,
                query: """
                    SELECT timestamp, open, high, low, close, volume
                    FROM candles
                    WHERE product_id = $1 AND granularity = $2
                      AND timestamp BETWEEN $3 AND $4
                    ORDER BY timestamp
                """
            )
            try stmt.bind(productId, at: 1)
            try stmt.bind(granularity, at: 2)
            try stmt.bind(Timestamp(start), at: 3)
            try stmt.bind(Timestamp(end), at: 4)
            let result = try stmt.execute()

            let timestamps = Array(result[0].cast(to: Timestamp.self)).compactMap { $0 }
            let opens = Array(result[1].cast(to: Double.self)).compactMap { $0 }
            let highs = Array(result[2].cast(to: Double.self)).compactMap { $0 }
            let lows = Array(result[3].cast(to: Double.self)).compactMap { $0 }
            let closes = Array(result[4].cast(to: Double.self)).compactMap { $0 }
            let volumes = Array(result[5].cast(to: Double.self)).compactMap { $0 }

            var candles: [DBCandle] = []
            for i in 0..<timestamps.count {
                guard i < opens.count, i < highs.count, i < lows.count,
                      i < closes.count, i < volumes.count else { break }
                candles.append(DBCandle(
                    timestamp: Foundation.Date(timestamps[i]),
                    open: opens[i], high: highs[i], low: lows[i],
                    close: closes[i], volume: volumes[i]
                ))
            }
            return candles
        }
    }

    // MARK: - Save candles via batch INSERT OR IGNORE (matches Python save_candles)

    public func saveCandles(
        _ candles: [DBCandle],
        productId: String,
        granularity: String = "300"
    ) throws {
        guard !candles.isEmpty else { return }
        try queue.sync {
            let stmt = try PreparedStatement(
                connection: connection,
                query: """
                    INSERT OR IGNORE INTO candles
                        (product_id, timestamp, open, high, low, close, volume, granularity)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """
            )
            for candle in candles {
                try stmt.bind(productId, at: 1)
                try stmt.bind(Timestamp(candle.timestamp), at: 2)
                try stmt.bind(candle.open, at: 3)
                try stmt.bind(candle.high, at: 4)
                try stmt.bind(candle.low, at: 5)
                try stmt.bind(candle.close, at: 6)
                try stmt.bind(candle.volume, at: 7)
                try stmt.bind(granularity, at: 8)
                let _ = try stmt.execute()
            }
        }
    }

    // MARK: - Edge params (matches Python save_edge_params / load_edge_params)

    public func saveEdgeParams(edge: String, params: EdgeParams) throws {
        try queue.sync {
            let stmt = try PreparedStatement(
                connection: connection,
                query: """
                    INSERT INTO edge_params (edge, curvature, y_depth, x_pixels, fee_rate, updated_at)
                    VALUES ($1, $2, $3, $4, $5, now())
                    ON CONFLICT (edge) DO UPDATE SET
                        curvature = excluded.curvature,
                        y_depth = excluded.y_depth,
                        x_pixels = excluded.x_pixels,
                        fee_rate = excluded.fee_rate,
                        updated_at = excluded.updated_at
                """
            )
            try stmt.bind(edge, at: 1)
            try stmt.bind(params.curvature, at: 2)
            try stmt.bind(Int32(params.yDepth), at: 3)
            try stmt.bind(Int32(params.xPixels), at: 4)
            try stmt.bind(params.feeRate, at: 5)
            let _ = try stmt.execute()
        }
    }

    public func loadEdgeParams() throws -> [String: EdgeParams] {
        try queue.sync {
            let result = try connection.query(
                "SELECT edge, curvature, y_depth, x_pixels, fee_rate FROM edge_params"
            )
            let edges = Array(result[0].cast(to: String.self)).compactMap { $0 }
            let curvatures = Array(result[1].cast(to: Double.self)).compactMap { $0 }
            let yDepths = Array(result[2].cast(to: Int32.self)).compactMap { $0 }
            let xPixels = Array(result[3].cast(to: Int32.self)).compactMap { $0 }
            let feeRates = Array(result[4].cast(to: Double.self)).compactMap { $0 }

            var params: [String: EdgeParams] = [:]
            for i in 0..<edges.count {
                guard i < curvatures.count, i < yDepths.count,
                      i < xPixels.count, i < feeRates.count else { break }
                params[edges[i]] = EdgeParams(
                    curvature: curvatures[i],
                    yDepth: Int(yDepths[i]),
                    xPixels: Int(xPixels[i]),
                    feeRate: feeRates[i]
                )
            }
            return params
        }
    }
}
