//
//  CoinGraph.swift
//  Autotrade
//
//  Graph structure for trading pairs with candle data.
//  Replaces pandas with native Swift collections.
//

import Foundation

// MARK: - Core Types

public struct EdgeState: Sendable {
    public var velocity: Double = 0.0
    public var ptt: Double = 0.0  // Upper band = profit target
    public var stop: Double = 0.0  // Lower band = stop loss
    public var hitPtt: Bool = false
    public var hitStop: Bool = false

    public init() {}
}

public struct NodeState: Sendable {
    public var height: Double = 0.0

    public init() {}
}

public struct Candle: Sendable, Codable {
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

    public var range: Double {
        high - low
    }

    public var body: Double {
        abs(close - open)
    }
}

// MARK: - Coin Graph

public final class CoinGraph: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.autotrade.coingraph", attributes: .concurrent)
    public let feeRate: Double

    public private(set) var nodes: Set<String> = []
    public private(set) var allPairs: [String] = []
    public private(set) var edges: [String: [Candle]] = [:]  // edge_id → candles
    public private(set) var edgeState: [String: EdgeState] = [:]
    public private(set) var nodeState: [String: NodeState] = [:]
    public private(set) var commonTimestamps: [Date] = []

    private var volatility: [String: [Double]] = [:]
    private let volWindow = 20

    private let cache = CandleCache()

    public init(feeRate: Double = 0.001) {
        self.feeRate = feeRate
    }

    // MARK: - Loading

    public func load(
        dbPath: String? = nil,
        minPartners: Int = 5,
        maxPartners: Int? = nil,
        lookbackDays: Int = 365,
        refreshBag: Bool = false,
        exchange: String = "coinbase",
        skipFetch: Bool = false
    ) async throws -> Int {
        // Load pairs from cache or API
        let pairs = try await loadPairs(
            exchange: exchange,
            minPartners: minPartners,
            maxPartners: maxPartners,
            refreshBag: refreshBag,
            skipFetch: skipFetch
        )

        self.allPairs = pairs
        self.nodes = extractNodes(from: pairs)

        // Initialize edge states
        for pair in pairs {
            edges[pair] = []
            edgeState[pair] = EdgeState()
        }

        // Initialize node states
        for node in nodes {
            nodeState[node] = NodeState()
        }

        // Load candle data
        try await loadCandles(
            dbPath: dbPath,
            pairs: pairs,
            lookbackDays: lookbackDays,
            skipFetch: skipFetch
        )

        // Align timestamps
        alignTimestamps()

        return pairs.count
    }

    private func extractNodes(from pairs: [String]) -> Set<String> {
        var nodes = Set<String>()
        for pair in pairs {
            let parts = pair.split(separator: "-")
            if parts.count == 2 {
                nodes.insert(String(parts[0]))
                nodes.insert(String(parts[1]))
            }
        }
        return nodes
    }

    private func loadPairs(
        exchange: String,
        minPartners: Int,
        maxPartners: Int?,
        refreshBag: Bool,
        skipFetch: Bool
    ) async throws -> [String] {
        // Try loading from bag cache first
        if !refreshBag {
            if let cached = try? loadBagCache() {
                print("Loaded bag of \(cached.count) pairs from cache")
                return cached
            }
        }

        if skipFetch {
            return []
        }

        // Build adjacency graph from available pairs
        let adjacency = try await buildAdjacencyGraph(exchange: exchange)

        // Filter by partner count
        let validCoins = adjacency.filter { _, partners in
            partners.count >= minPartners &&
            (maxPartners == nil || partners.count <= maxPartners!)
        }

        // Generate pair list
        var pairs: [String] = []
        for (coin, partners) in validCoins {
            for partner in partners {
                let pair1 = "\(coin)-\(partner)"
                _ = "\(partner)-\(coin)"

                // Only add if coin < partner (avoid duplicates)
                if coin < partner {
                    pairs.append(pair1)
                }
            }
        }

        // Save to cache
        try? saveBagCache(pairs)

        return pairs
    }

    private func buildAdjacencyGraph(exchange: String) async throws -> [String: Set<String>] {
        // Use DuckDB to get unique products from database
        var adjacency: [String: Set<String>] = [:]

        if let products = try? cache.fetchProducts() {
            for product in products {
                let parts = product.split(separator: "-")
                guard parts.count == 2 else { continue }

                let base = String(parts[0])
                let quote = String(parts[1])

                adjacency[base, default: []].insert(quote)
                adjacency[quote, default: []].insert(base)
            }
        }

        return adjacency
    }

    private func loadCandles(
        dbPath: String?,
        pairs: [String],
        lookbackDays: Int,
        skipFetch: Bool
    ) async throws {
        let endDate = Date()
        let startDate = endDate.addingTimeInterval(-Double(lookbackDays) * 24 * 3600)

        for pair in pairs {
            if let candles = try? await cache.fetchCandles(
                product: pair,
                start: startDate,
                end: endDate
            ) {
                edges[pair] = candles
            }
        }
    }

    private func alignTimestamps() {
        // Find intersection of all timestamps
        guard !edges.isEmpty else { return }

        var timestampSets: [Set<Date>] = []
        for candles in edges.values {
            if !candles.isEmpty {
                let timestamps = Set(candles.map { $0.timestamp })
                timestampSets.append(timestamps)
            }
        }

        // Find common timestamps
        guard !timestampSets.isEmpty else { return }
        var common = timestampSets[0]
        for set in timestampSets.dropFirst() {
            common.formIntersection(set)
        }

        commonTimestamps = common.sorted()
    }

    // MARK: - Updates

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

        let timestamp = commonTimestamps[barIdx]

        for (pair, candles) in edges {
            guard let idx = candles.firstIndex(where: { $0.timestamp == timestamp }) else {
                continue
            }

            let candle = candles[idx]
            let range = candle.range

            // Update velocity (simple momentum)
            let prevVel = edgeState[pair]?.velocity ?? 0.0
            let accel = range - prevVel
            let newVel = prevVel * 0.9 + accel * 0.1  // Exponential smoothing

            edgeVelocities[pair] = newVel
            edgeAccels[pair] = accel

            // Check bands
            if let state = edgeState[pair] {
                hitPtt[pair] = candle.high >= state.ptt
                hitStop[pair] = candle.low <= state.stop

                // Update state
                var newState = state
                newState.velocity = newVel
                newState.hitPtt = hitPtt[pair] ?? false
                newState.hitStop = hitStop[pair] ?? false
                edgeState[pair] = newState
            }

            // Update volatility
            if volatility[pair] == nil {
                volatility[pair] = []
            }
            volatility[pair]?.append(range)
            if volatility[pair]?.count ?? 0 > volWindow {
                volatility[pair]?.removeFirst()
            }
        }

        return (edgeAccels, edgeVelocities, hitPtt, hitStop)
    }

    // MARK: - Cache

    private func loadBagCache() throws -> [String] {
        let path = bagCachePath()
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode([String].self, from: data)
    }

    private func saveBagCache(_ pairs: [String]) throws {
        let path = bagCachePath()
        let data = try JSONEncoder().encode(pairs)
        try data.write(to: path)
    }

    private func bagCachePath() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let docsDir = paths[0]
        return docsDir.appendingPathComponent("pair_bag.json")
    }

    public func getVolatility(pair: String) -> Double {
        guard let vols = volatility[pair], !vols.isEmpty else {
            return 0.0
        }
        return vols.reduce(0, +) / Double(vols.count)
    }
}

// MARK: - Candle Cache

public final class CandleCache: @unchecked Sendable {
    private let dbPath: String

    public init(dbPath: String = "candles.duckdb") {
        self.dbPath = dbPath
    }

    public func fetchProducts() throws -> [String] {
        // Use DuckDB to fetch unique products
        // This would typically use a Swift SQL client or system() call
        return []
    }

    public func fetchCandles(
        product: String,
        start: Date,
        end: Date
    ) async throws -> [Candle] {
        // Use DuckDB to fetch candles
        // This would typically use a Swift SQL client or system() call
        return []
    }

    public func saveCandles(_ candles: [Candle], for product: String) throws {
        // Save candles to DuckDB
        // This would typically use a Swift SQL client or system() call
    }
}
