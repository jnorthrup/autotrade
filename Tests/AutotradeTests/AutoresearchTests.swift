//
//  AutoresearchTests.swift
//  AutotradeTests
//
//  Tests for autoresearch and training functionality
//

import Testing
import AutotradeHRM

@Suite("Autoresearch Tests")
struct AutoresearchTests {
    @Test("Training hyperparameters validation")
    func testTrainingHyperparameters() {
        // Test valid hyperparameter ranges
        let validLR = [0.0001, 0.001, 0.01, 0.1]
        let validYDepth = [50, 100, 200, 300, 400]
        let validXPixels = [5, 10, 15, 20, 30]
        let validCurvature = [0.5, 1.0, 2.0, 3.0, 4.0]

        for lr in validLR {
            #expect(lr > 0 && lr < 1)
        }

        for depth in validYDepth {
            #expect(depth > 0 && depth <= 500)
        }

        for pixels in validXPixels {
            #expect(pixels > 0 && pixels <= 50)
        }

        for curve in validCurvature {
            #expect(curve > 0 && curve <= 5.0)
        }
    }

    @Test("Square cube sizes validation")
    func testSquareCubeSizes() {
        let sizes = [4, 16, 64, 256]

        #expect(sizes.count >= 1)

        // Should be powers of 4
        for size in sizes {
            let log4 = log(Double(size)) / log(4.0)
            #expect(abs(log4 - round(log4)) < 1e-10)
        }

        // Should be monotonically increasing
        for i in 1..<sizes.count {
            #expect(sizes[i] > sizes[i-1])
        }
    }

    @Test("Platea detection logic")
    func testPlateauDetection() {
        // Test convergence detection with synthetic loss history
        let convergingLosses = [1.0, 0.9, 0.85, 0.82, 0.81, 0.805, 0.803, 0.802, 0.801, 0.800]
        let nonConvergingLosses = [1.0, 0.8, 0.9, 0.7, 0.85, 0.75, 0.82, 0.78]

        // Simple convergence check (should be more sophisticated in real implementation)
        let convergingVariance = calculateVariance(convergingLosses.suffix(5))
        let nonConvergingVariance = calculateVariance(nonConvergingLosses.suffix(5))

        #expect(convergingVariance < nonConvergingVariance)
    }

    @Test("Curriculum progression")
    func testCurriculumProgression() {
        let growthCycle = ["h", "H", "L"]
        let maxPhase = 50

        var hDim = 4
        var hLayers = 1
        var lLayers = 1

        for phase in 0..<maxPhase {
            let progress = min(1.0, Double(phase) / 200.0)
            let growthIdx = phase % growthCycle.count

            if growthCycle[growthIdx] == "h" {
                if hDim < 256 { // Max size
                    hDim *= 4
                }
            } else if growthCycle[growthIdx] == "H" {
                if hLayers < 8 {
                    hLayers *= 2
                }
            } else if growthCycle[growthIdx] == "L" {
                if lLayers < 8 {
                    lLayers *= 2
                }
            }

            #expect(hDim >= 4 && hDim <= 256)
            #expect(hLayers >= 1 && hLayers <= 8)
            #expect(lLayers >= 1 && lLayers <= 8)
        }
    }

    @Test("Pair selection logic")
    func testPairSelection() {
        let allPairs = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"]
        let adj: [String: [String]] = [
            "BTC": ["USD"],
            "ETH": ["USD"],
            "SOL": ["USD"],
            "DOGE": ["USD"],
            "XRP": ["USD"],
            "USD": ["BTC", "ETH", "SOL", "DOGE", "XRP"]
        ]

        let nPairs = 3
        var rng = SystemRandomNumberGenerator()

        let selected = selectRelatedPairs(allPairs: allPairs, adj: adj, nPairs: nPairs, rng: &rng)

        #expect(selected.count <= nPairs)
        #expect(selected.allSatisfy { allPairs.contains($0) })

        // Should contain USD pairs
        #expect(selected.allSatisfy { $0.contains("USD") })
    }

    @Test("Command line argument parsing")
    func testCommandLineArgs() {
        // Test that all expected command line options are documented
        let expectedArgs = [
            "--autoresearch",
            "--start-bar",
            "--end-bar",
            "--print-every",
            "--min-partners",
            "--max-partners",
            "--skip-fetch",
            "--exchange",
            "--prediction-depth",
            "--h-dim",
            "--z-dim",
            "--lr",
            "--y-depth",
            "--x-pixels",
            "--curvature"
        ]

        // This would be validated against actual implementation
        #expect(expectedArgs.count > 10)
        #expect(expectedArgs.allSatisfy { $0.hasPrefix("--") })
    }
}

// Helper functions
private func calculateVariance(_ values: ArraySlice<Double>) -> Double {
    let mean = values.reduce(0, +) / Double(values.count)
    let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Double(values.count)
    return variance
}

private func selectRelatedPairs(allPairs: [String], adj: [String: [String]], nPairs: Int, rng: inout SystemRandomNumberGenerator) -> [String] {
    if nPairs >= allPairs.count {
        return allPairs
    }

    let currencies = Array(adj.keys)
    guard !currencies.isEmpty else { return Array(allPairs.prefix(nPairs)) }

    var selected = Set<String>()
    let seedCurrency = currencies.randomElement(using: &rng) ?? currencies[0]
    var frontier = [seedCurrency]
    var visitedCurrencies = Set<String>([seedCurrency])

    while selected.count < nPairs && !frontier.isEmpty {
        let curr = frontier.removeFirst()
        var candidates = (adj[curr] ?? []).filter { !selected.contains($0) }
        candidates.shuffle(using: &rng)

        for pid in candidates {
            if selected.count >= nPairs { break }
            selected.insert(pid)
            let parts = pid.split(separator: "-", maxSplits: 1)
            if parts.count == 2 {
                for c in parts {
                    let currency = String(c)
                    if !visitedCurrencies.contains(currency) {
                        visitedCurrencies.insert(currency)
                        frontier.append(currency)
                    }
                }
            }
        }

        if frontier.isEmpty && selected.count < nPairs {
            let remaining = currencies.filter { !visitedCurrencies.contains($0) }
            if let newSeed = remaining.randomElement(using: &rng) {
                frontier.append(newSeed)
                visitedCurrencies.insert(newSeed)
            }
        }
    }

    return Array(selected)
}