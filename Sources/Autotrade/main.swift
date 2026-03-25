//
//  main.swift
//  Autotrade
//
//  Simple test version
//

import Foundation
import AutotradeHRM

@main
struct AutotradeApp {
    static func main() async {
        Swift.print("🚀 Autotrade Swift (ANE-accelerated)")
        Swift.print("Exchange: coinbase, Min partners: 10")

        let graph = CoinGraph(feeRate: 0.001)

        do {
            Swift.print("Loading coin graph...")
            let nBars = try await graph.load(minPartners: 10, exchange: "coinbase", skipFetch: false)
            Swift.print("Loaded \(graph.nodes.count) nodes, \(graph.edges.count) edges, \(nBars) bars")

            if nBars == 0 {
                Swift.print("No data available")
                return
            }

            // Test single pair loading
            if let firstPair = graph.allPairs.first {
                Swift.print("Testing single pair load for \(firstPair)...")
                let startDate = Date().addingTimeInterval(-365*24*3600)
                let endDate = Date()
                let candles = try graph.db.queryCandles(product: firstPair, start: startDate, end: endDate)
                Swift.print("✅ SUCCESS: Loaded \(candles.count) candles for \(firstPair)")
            }

            Swift.print("🎯 Swift transcription working!")

        } catch {
            Swift.print("❌ Error: \(error)")
        }
    }
}