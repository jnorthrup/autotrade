import Foundation
import DuckDB

public final class DuckDBFFI: @unchecked Sendable {
    private let database: Database
    private let connection: Connection

    public init(path: String = "candles.duckdb") throws {
        database = try Database(store: .file(at: URL(fileURLWithPath: path)))
        connection = try database.connect()
    }

    public func queryProducts() throws -> [String] {
        let result = try connection.query("SELECT DISTINCT product_id FROM candles")
        let column = result[0].cast(to: String.self)
        return Array(column).compactMap { $0 }
    }

    public func queryAllCandles(pairs: [String], start: Foundation.Date, end: Foundation.Date) throws -> [String: [DBCandle]] {
        let df = ISO8601DateFormatter()
        df.formatOptions = [.withInternetDateTime]

        let startStr = df.string(from: start)
        let endStr = df.string(from: end)

        var result: [String: [DBCandle]] = [:]

        // Single massive query with IN clause - much faster than per-pair queries
        let pairsList = pairs.map { "'\($0)'" }.joined(separator: ", ")
        let sql = """
        SELECT product_id, timestamp, open, high, low, close, volume
        FROM candles
        WHERE product_id IN (\(pairsList))
          AND timestamp >= '\(startStr)'
          AND timestamp <= '\(endStr)'
        ORDER BY product_id, timestamp
        """

        let allResults = try connection.query(sql)

        // Access columns by index using official DuckDB Swift API
        let productIdsArray = Array(allResults[0].cast(to: String.self))
        let opensArray = Array(allResults[2].cast(to: Double.self))
        let highsArray = Array(allResults[3].cast(to: Double.self))
        let lowsArray = Array(allResults[4].cast(to: Double.self))
        let closesArray = Array(allResults[5].cast(to: Double.self))
        let volumesArray = Array(allResults[6].cast(to: Double.self))

        // Get timestamp column once
        let timestampsColumn = allResults[1].cast(to: Timestamp.self)

        // Iterate through arrays and group by product_id
        for i in 0..<productIdsArray.count {
            let idx = UInt64(i)
            guard let productId = productIdsArray[i],
                  let ts = timestampsColumn[idx],
                  let open = opensArray[i],
                  let high = highsArray[i],
                  let low = lowsArray[i],
                  let close = closesArray[i],
                  let volume = volumesArray[i] else {
                continue
            }

            // Timestamp stores micros since epoch - convert to Date
            let date = Foundation.Date(timeIntervalSince1970: TimeInterval(ts.microseconds) / 1_000_000)

            let candle = DBCandle(
                timestamp: date,
                open: open,
                high: high,
                low: low,
                close: close,
                volume: volume
            )
            result[productId, default: []].append(candle)
        }

        return result
    }

    public func queryCandles(product: String, start: Foundation.Date, end: Foundation.Date) throws -> [DBCandle] {
        let df = ISO8601DateFormatter()
        df.formatOptions = [.withInternetDateTime]

        let startStr = df.string(from: start)
        let endStr = df.string(from: end)

        let sql = """
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE product_id = '\(product)'
          AND timestamp >= '\(startStr)'
          AND timestamp <= '\(endStr)'
        ORDER BY timestamp
        """

        let result = try connection.query(sql)
        let timestamps = Array(result[0].cast(to: Foundation.Date.self)).compactMap { $0 }
        let opens = Array(result[1].cast(to: Double.self)).compactMap { $0 }
        let highs = Array(result[2].cast(to: Double.self)).compactMap { $0 }
        let lows = Array(result[3].cast(to: Double.self)).compactMap { $0 }
        let closes = Array(result[4].cast(to: Double.self)).compactMap { $0 }
        let volumes = Array(result[5].cast(to: Double.self)).compactMap { $0 }

        var candles: [DBCandle] = []
        for i in 0..<timestamps.count {
            let date = timestamps[i]
            let open = opens[i]
            let high = highs[i]
            let low = lows[i]
            let close = closes[i]
            let volume = volumes[i]

            candles.append(DBCandle(timestamp: date, open: open, high: high, low: low, close: close, volume: volume))
        }

        return candles
    }
}