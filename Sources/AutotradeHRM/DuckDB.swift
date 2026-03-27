//
//  DuckDB.swift
//  AutotradeHRM
//
//  DuckDB FFI wrapper
//

import Foundation

public struct DBCandle: Sendable, Codable {
    public let timestamp: Date
    public let open: Double
    public let high: Double
    public let low: Double
    public let close: Double
    public let volume: Double
    
    public init(timestamp: Date, open: Double, high: Double, low: Double, close: Double, volume: Double) {
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    }
}

public final class DuckDB: @unchecked Sendable {
    private let dbPath: String
    private let queue = DispatchQueue(label: "com.autotrade.duckdb")
    
    public init(path: String = "candles.duckdb") {
        self.dbPath = path
    }
    
    public func queryProducts() throws -> [String] {
        let sql = "SELECT DISTINCT product_id FROM candles"
        return try runQuery(sql).map { $0[0] }
    }
    
    public func queryCandles(product: String, start: Date, end: Date) throws -> [DBCandle] {
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
        
        let rows = try runQuery(sql)
        return rows.compactMap { row -> DBCandle? in
            guard row.count >= 6,
                  let ts = df.date(from: row[0]),
                  let open = Double(row[1]),
                  let high = Double(row[2]),
                  let low = Double(row[3]),
                  let close = Double(row[4]),
                  let volume = Double(row[5]) else {
                return nil
            }
            return DBCandle(timestamp: ts, open: open, high: high, low: low, close: close, volume: volume)
        }
    }
    
    public func execute(_ sql: String) throws {
        try queue.sync {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            task.arguments = ["duckdb", dbPath, "-cmd", sql]
            task.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            
            let outPipe = Pipe()
            let errPipe = Pipe()
            task.standardOutput = outPipe
            task.standardError = errPipe
            
            try task.run()
            task.waitUntilExit()
            
            guard task.terminationStatus == 0 else {
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                let errMsg = String(data: errData, encoding: .utf8) ?? "Unknown error"
                throw NSError(domain: "DuckDB", code: Int(task.terminationStatus), userInfo: [NSLocalizedDescriptionKey: errMsg])
            }
        }
    }
    
    private func runQuery(_ sql: String) throws -> [[String]] {
        return try queue.sync {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            task.arguments = ["duckdb", dbPath, "-csv", "-cmd", sql]
            task.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            
            let outPipe = Pipe()
            let errPipe = Pipe()
            task.standardOutput = outPipe
            task.standardError = errPipe
            
            try task.run()
            task.waitUntilExit()
            
            guard task.terminationStatus == 0 else {
                throw NSError(domain: "DuckDB", code: Int(task.terminationStatus), userInfo: nil)
            }
            
            let data = outPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? ""
            
            var results: [[String]] = []
            var lines = output.split(separator: "\n", omittingEmptySubsequences: true)
            
            if !lines.isEmpty {
                lines.removeFirst()
            }
            
            for line in lines {
                let parts = line.split(separator: ",", omittingEmptySubsequences: false).map {
                    $0.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                }
                results.append(parts)
            }
            
            return results
        }
    }
}
