//
//  CoinGraph.swift
//  Autotrade
//
//  Port of Python coin_graph.py
//

import Foundation
import AutotradeHRM

public struct EdgeState: Sendable {
    public var velocity: Double = 0.0
    public var ptt: Double = 0.0
    public var stop: Double = 0.0
    public var hitPtt: Bool = false
    public var hitStop: Bool = false
    
    public init() {}
}

public struct NodeState: Sendable {
    public var height: Double = 0.0
    
    public init() {}
}

public class CoinGraph: @unchecked Sendable {
    public let feeRate: Double
    public private(set) var nodes: Set<String> = []
    public private(set) var allPairs: [String] = []
    public private(set) var edges: [String: [DBCandle]] = [:]
    public private(set) var edgeState: [String: EdgeState] = [:]
    public private(set) var nodeState: [String: NodeState] = [:]
    public private(set) var commonTimestamps: [Date] = []
    
    private var volatility: [String: RollingMeanBuffer] = [:]
    // Keep a per-edge cursor so sequential bar traversal advances instead of rescanning.
    private var edgeCursorState: [String: EdgeCursorState] = [:]
    private let volWindow = 20
    private var db: DuckDBFFI?
    
    public init(feeRate: Double = 0.001) {
        self.feeRate = feeRate
    }
    
    public func load(minPartners: Int = 5, exchange: String = "coinbase", skipFetch: Bool = false, bagPath: URL? = nil) async throws -> Int {
        Swift.print("DEBUG: load() method ENTERED")
        Swift.print("Loading coin graph...")
        Swift.print("DEBUG: Initializing DB...")

        Swift.print("DEBUG: Creating DuckDBFFI...")
        db = try DuckDBFFI(path: "candles.duckdb")
        Swift.print("DEBUG: DuckDBFFI assigned")
        guard let db = db else {
            Swift.print("DEBUG: db is nil!")
            return 0
        }
        Swift.print("DEBUG: DB initialized successfully")
        
        let bagUrl = bagPath ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("pair_bag.json")
        
        var pairs: [String] = []
        Swift.print("DEBUG: Checking cache at \(bagUrl.path)...")

        // Load from cache
        if !skipFetch, let data = try? Data(contentsOf: bagUrl),
           let cached = try? JSONDecoder().decode([String].self, from: data) {
            pairs = cached
            Swift.print("Loaded bag of \(pairs.count) pairs from cache")
        } else {
            Swift.print("DEBUG: Cache miss, discovering pairs...")
        }
        
        // Discover pairs from database if not cached
        if pairs.isEmpty {
            let products = try db.queryProducts()
            var adjacency: [String: Set<String>] = [:]
            
            for product in products {
                let parts = product.split(separator: "-", maxSplits: 1)
                guard parts.count == 2 else { continue }
                let base = String(parts[0])
                let quote = String(parts[1])
                adjacency[base, default: []].insert(quote)
                adjacency[quote, default: []].insert(base)
            }
            
            let coinSet = Set(adjacency.filter { $0.value.count >= minPartners }.keys)
            
            let fiatExclude: Set<String> = ["GBP", "EUR", "SGD"]
            let usdBases = Set<String>(products.lazy.filter { $0.hasSuffix("-USD") }.map { $0.split(separator: "-")[0] }.map { String($0) })
            
            var seen: Set<[String]> = []
            
            for pid in products {
                let parts = pid.split(separator: "-", maxSplits: 1)
                guard parts.count == 2 else { continue }
                let base = String(parts[0])
                let quote = String(parts[1])
                
                if fiatExclude.contains(String(quote)) || fiatExclude.contains(String(base)) { continue }
                if (String(quote) == "USDC" || String(quote) == "USDT") && usdBases.contains(String(base)) { continue }
                
                if coinSet.contains(base) && coinSet.contains(quote) {
                    let canonical = [base, quote].sorted()
                    if !seen.contains(canonical) {
                        seen.insert(canonical)
                        pairs.append(pid)
                    }
                }
            }
            
            if let data = try? JSONEncoder().encode(pairs) {
                try? data.write(to: bagUrl)
                Swift.print("Saved bag of \(pairs.count) pairs")
            }
        }
        
        self.allPairs = pairs
        Swift.print("Graph discovery: \(pairs.count) pairs")
        
        // Load ALL candles in one massive query for Python-like performance
        let endDate = Date()
        let startDate = endDate.addingTimeInterval(-365 * 24 * 3600)

        let start = Date()
        Swift.print("  Loading candles for \(pairs.count) pairs...")
        Swift.print("  Date range: \(startDate) to \(endDate)")

        // Load in batches of 20 pairs to avoid SQL IN clause limits
        let batchSize = 20
        var candlesByPair: [String: [DBCandle]] = [:]
        candlesByPair.reserveCapacity(pairs.count)

        for batchStart in stride(from: 0, to: pairs.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, pairs.count)
            let batch = Array(pairs[batchStart..<batchEnd])

            Swift.print("  Loading batch \(batchStart/batchSize + 1)/\((pairs.count + batchSize - 1)/batchSize) with \(batch.count) pairs...")
            let batchCandles = try db.queryAllCandles(pairs: batch, start: startDate, end: endDate)

            for (product, candles) in batchCandles {
                candlesByPair[product] = candles
            }
        }

        // Assign to edges and track loaded pairs
        var loaded = 0
        edges.removeAll(keepingCapacity: true)
        edgeState.removeAll(keepingCapacity: true)
        nodeState.removeAll(keepingCapacity: true)
        volatility.removeAll(keepingCapacity: true)
        edgeCursorState.removeAll(keepingCapacity: true)
        edges.reserveCapacity(pairs.count)
        edgeState.reserveCapacity(pairs.count * 2)
        nodeState.reserveCapacity(pairs.count * 2)
        volatility.reserveCapacity(pairs.count)
        edgeCursorState.reserveCapacity(pairs.count)
        for product in pairs {
            if let candles = candlesByPair[product], !candles.isEmpty {
                edges[product] = candles
                resetEdgeCursor(edge: product)

                let parts = product.split(separator: "-", maxSplits: 1)
                if parts.count == 2 {
                    let base = String(parts[0])
                    let quote = String(parts[1])
                    nodes.insert(base)
                    nodes.insert(quote)
                }
                edgeState[product] = EdgeState()
                loaded += 1

                if loaded % 20 == 0 {
                    Swift.print("  Loaded \(loaded)/\(pairs.count) pairs (\(String(format: "%.1f", Date().timeIntervalSince(start)))s)")
                    fflush(stdout)
                }
            }
        }
        
        Swift.print("  Loaded \(loaded) pairs in \(String(format: "%.1f", Date().timeIntervalSince(start)))s")
        
        // Init edge states for reverse pairs and node states
        for product in pairs {
            let parts = product.split(separator: "-", maxSplits: 1)
            if parts.count == 2 {
                let base = String(parts[0])
                let quote = String(parts[1])
                edgeState[base + "-" + quote] = EdgeState()
                edgeState[quote + "-" + base] = EdgeState()
                nodeState[base] = NodeState()
                nodeState[quote] = NodeState()
            }
        }
        
        // Always put USD first
        if nodes.contains("USD") {
            nodes.remove("USD")
            let sorted = nodes.sorted()
            nodes = ["USD"]
            nodes.formUnion(sorted)
        }
        
        alignTimestamps()
        return commonTimestamps.count
    }
    
    private func alignTimestamps() {
        if edges.isEmpty { return }
        
        var allIndices: [Set<Date>] = []
        for candles in edges.values {
            allIndices.append(Set(candles.map { $0.timestamp }))
        }
        
        if allIndices.isEmpty { return }
        
        var common = allIndices[0]
        for idx in allIndices.dropFirst() {
            common = common.union(idx)
        }
        
        commonTimestamps = common.sorted()
        Swift.print("Aligned \(commonTimestamps.count) bars across \(nodes.count) nodes")
    }
    
    public func update(barIdx: Int) -> (
        edgeAccels: [String: Double],
        edgeVelocities: [String: Double],
        hitPtt: [String: Bool],
        hitStop: [String: Bool]
    ) {
        var edgeAccels: [String: Double] = [:]
        var edgeVelocities: [String: Double] = [:]
        var hitPtt: [String: Bool] = [:]
        var hitStop: [String: Bool] = [:]
        
        guard barIdx < commonTimestamps.count else {
            return (edgeAccels, edgeVelocities, hitPtt, hitStop)
        }
        
        let ts = commonTimestamps[barIdx]
        
        for (product, _) in edges {
            guard let candle = candle(at: ts, for: product) else { continue }
            let close = candle.close
            let openPrice = candle.open
            
            var velocity = 0.0
            if openPrice > 0 {
                velocity = log(close / openPrice)
            }
            
            let prevVelocity = edgeState[product]?.velocity ?? 0.0
            let accel = velocity - prevVelocity
            
            edgeState[product]?.velocity = velocity
            edgeAccels[product] = accel
            edgeVelocities[product] = velocity
            
            // Track volatility with a fixed-size rolling mean instead of shifting arrays every bar.
            var rollingVolatility = volatility[product] ?? RollingMeanBuffer(limit: volWindow)
            let vol = rollingVolatility.appendAndMean(abs(velocity))
            volatility[product] = rollingVolatility
            edgeState[product]?.ptt = feeRate + vol
            edgeState[product]?.stop = -(feeRate + vol)
            
            // Check band crossings
            let pttVal = edgeState[product]?.ptt ?? 0
            let stopVal = edgeState[product]?.stop ?? 0
            hitPtt[product] = velocity > pttVal
            hitStop[product] = velocity < stopVal
            edgeState[product]?.hitPtt = hitPtt[product] ?? false
            edgeState[product]?.hitStop = hitStop[product] ?? false
        }
        
        computeHeights(edgeAccels: edgeAccels)
        return (edgeAccels, edgeVelocities, hitPtt, hitStop)
    }
    
    public func setEdges(
        edges: [String: [DBCandle]],
        edgeState: [String: EdgeState],
        nodeState: [String: NodeState],
        nodes: Set<String>,
        allPairs: [String],
        commonTimestamps: [Date]
    ) {
        self.edges = edges
        self.edgeState = edgeState
        self.nodeState = nodeState
        self.nodes = nodes
        self.allPairs = allPairs
        self.commonTimestamps = commonTimestamps
        volatility.removeAll(keepingCapacity: true)
        edgeCursorState.removeAll(keepingCapacity: true)
        edgeCursorState.reserveCapacity(edges.count)
        for edge in edges.keys {
            resetEdgeCursor(edge: edge)
        }
    }
    
    private struct EdgeCursorState {
        var lastTimestamp: Date?
        var lastIndex: Int
    }

    private struct RollingMeanBuffer {
        let limit: Int
        var values: [Double]
        var start: Int
        var count: Int
        var sum: Double

        init(limit: Int) {
            self.limit = max(1, limit)
            self.values = Array(repeating: 0.0, count: max(1, limit))
            self.start = 0
            self.count = 0
            self.sum = 0.0
        }

        mutating func appendAndMean(_ value: Double) -> Double {
            if count < limit {
                let insertIndex = (start + count) % limit
                values[insertIndex] = value
                count += 1
                sum += value
            } else {
                sum -= values[start]
                values[start] = value
                sum += value
                start = (start + 1) % limit
            }

            return count > 0 ? sum / Double(count) : 0.0
        }
    }

    private func resetEdgeCursor(edge: String) {
        edgeCursorState[edge] = EdgeCursorState(lastTimestamp: nil, lastIndex: 0)
        volatility[edge] = RollingMeanBuffer(limit: volWindow)
    }

    private func candleLowerBound(for timestamp: Date, in candles: [DBCandle]) -> Int {
        var lower = 0
        var upper = candles.count

        while lower < upper {
            let mid = lower + (upper - lower) / 2
            if candles[mid].timestamp < timestamp {
                lower = mid + 1
            } else {
                upper = mid
            }
        }

        return lower
    }

    private func candle(at timestamp: Date, for edge: String) -> DBCandle? {
        guard let candles = edges[edge], !candles.isEmpty else {
            return nil
        }

        if let cached = edgeCursorState[edge], let lastTimestamp = cached.lastTimestamp, timestamp >= lastTimestamp {
            var index = min(cached.lastIndex, candles.count - 1)

            if candles[index].timestamp < timestamp {
                while index + 1 < candles.count, candles[index].timestamp < timestamp {
                    index += 1
                }
            } else if candles[index].timestamp > timestamp {
                index = candleLowerBound(for: timestamp, in: candles)
            }

            if index < candles.count, candles[index].timestamp == timestamp {
                edgeCursorState[edge] = EdgeCursorState(lastTimestamp: timestamp, lastIndex: index)
                return candles[index]
            }

            edgeCursorState[edge] = EdgeCursorState(lastTimestamp: timestamp, lastIndex: min(index, candles.count - 1))
            return nil
        }

        let index = candleLowerBound(for: timestamp, in: candles)
        guard index < candles.count, candles[index].timestamp == timestamp else {
            edgeCursorState[edge] = EdgeCursorState(lastTimestamp: timestamp, lastIndex: min(index, candles.count - 1))
            return nil
        }

        edgeCursorState[edge] = EdgeCursorState(lastTimestamp: timestamp, lastIndex: index)
        return candles[index]
    }

    private func computeHeights(edgeAccels: [String: Double]) {
        var outflow: [String: [Double]] = [:]
        
        for (product, accel) in edgeAccels {
            let parts = product.split(separator: "-", maxSplits: 1)
            if parts.count == 2 {
                let base = String(parts[0])
                outflow[base, default: []].append(accel)
            }
        }
        
        for node in nodeState.keys {
            let values = outflow[node] ?? []
            nodeState[node]?.height = values.isEmpty ? 0.0 : values.reduce(0, +) / Double(values.count)
        }
    }
}
