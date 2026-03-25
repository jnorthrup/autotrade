//
//  DuckDBFFI.swift
//  AutotradeHRM
//
//  DuckDB Swift API wrapper - using official DuckDB Swift library
//

import Foundation
import DuckDB

public final class DuckDBFFI: @unchecked Sendable {
    private let database: Database
    private let connection: Connection

    public init(path: String = "candles.duckdb") {
        do {
            // Use file-based database for persistence
            self.database = try Database(store: .file(path: path))
            self.connection = try Connection(database: self.database)
        } catch {
            fatalError("Failed to initialize DuckDB: \(error)")
        }
    }

    public func queryProducts() throws -> [String] {
        let result = try connection.query("SELECT DISTINCT product_id FROM candles")
        return result.rows.compactMap { row in
            try? row[0].cast(to: String.self)
        }
    }

    public func queryAllCandles(pairs: [String], start: Date, end: Date) throws -> [String: [DBCandle]] {
        let df = ISO8601DateFormatter()
        df.formatOptions = [.withInternetDateTime]
        let startStr = df.string(from: start)
        let endStr = df.string(from: end)

        // Single massive query for ALL pairs using official DuckDB Swift API
        let pairList = pairs.map { "'\($0)'" }.joined(separator: ",")
        let sql = """
        SELECT product_id, timestamp, open, high, low, close, volume
        FROM candles
        WHERE product_id IN (\(pairList))
        AND timestamp >= '\(startStr)'
        AND timestamp <= '\(endStr)'
        ORDER BY product_id, timestamp
        """

        let result = try connection.query(sql)
        var candlesByPair: [String: [DBCandle]] = [:]

        for row in result.rows {
            guard let product = try? row[0].cast(to: String.self),
                  let timestampStr = try? row[1].cast(to: String.self),
                  let ts = df.date(from: timestampStr),
                  let open = try? row[2].cast(to: Double.self),
                  let high = try? row[3].cast(to: Double.self),
                  let low = try? row[4].cast(to: Double.self),
                  let close = try? row[5].cast(to: Double.self),
                  let volume = try? row[6].cast(to: Double.self) else {
                continue
            }

            let candle = DBCandle(timestamp: ts, open: open, high: high, low: low, close: close, volume: volume)
            candlesByPair[product, default: []].append(candle)
        }

        return candlesByPair
    }

    public func queryCandles(product: String, start: Date, end: Date) throws -> [DBCandle] {
        let df = ISO8601DateFormatter()
        df.formatOptions = [.withInternetDateTime]
        let startStr = df.string(from: start)
        let endStr = df.string(from: end)

        let sql = """
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE product_id = ?
        AND timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp
        """

        let result = try connection.query(sql, parameters: [product, startStr, endStr])
        var candles: [DBCandle] = []

        for row in result.rows {
            guard let timestampStr = try? row[0].cast(to: String.self),
                  let ts = df.date(from: timestampStr),
                  let open = try? row[1].cast(to: Double.self),
                  let high = try? row[2].cast(to: Double.self),
                  let low = try? row[3].cast(to: Double.self),
                  let close = try? row[4].cast(to: Double.self),
                  let volume = try? row[5].cast(to: Double.self) else {
                continue
            }

            let candle = DBCandle(timestamp: ts, open: open, high: high, low: low, close: close, volume: volume)
            candles.append(candle)
        }

        return candles
    }
}