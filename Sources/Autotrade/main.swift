//
//  main.swift
//  autotrade
//
//  Main entry point for Swift autotrade with ANE inference.
//

import Foundation
import AutotradeHRM

print("🚀 Autotrade Swift (ANE-accelerated)")

let arguments = CommandLine.arguments

// Parse arguments
var autoresearch = false
var exchange = "coinbase"
var minPartners = 5
var startBar = 0
var endBar: Int? = nil
var printEvery = 100

var i = 1
while i < arguments.count {
    switch arguments[i] {
    case "--autoresearch":
        autoresearch = true
    case "--exchange":
        i += 1
        if i < arguments.count {
            exchange = arguments[i]
        }
    case "--min-partners":
        i += 1
        if i < arguments.count {
            minPartners = Int(arguments[i]) ?? 5
        }
    case "--start-bar":
        i += 1
        if i < arguments.count {
            startBar = Int(arguments[i]) ?? 0
        }
    case "--end-bar":
        i += 1
        if i < arguments.count {
            endBar = Int(arguments[i])
        }
    case "--print-every":
        i += 1
        if i < arguments.count {
            printEvery = Int(arguments[i]) ?? 100
        }
    default:
        break
    }
    i += 1
}

print("Exchange: \(exchange), Min partners: \(minPartners)")
print("")

// Initialize graph
let graph = CoinGraph(feeRate: 0.001)

print("Loading trading pairs...")

Task {
    let pairCount = try await graph.load(
        minPartners: minPartners,
        exchange: exchange,
        skipFetch: false
    )

    print("Loaded \(pairCount) pairs with \(graph.commonTimestamps.count) bars")

    if autoresearch {
        try await runAutoresearch(graph: graph)
    } else {
        try await runTraining(graph: graph, startBar: startBar, endBar: endBar, printEvery: printEvery)
    }

    exit(0)
}

RunLoop.main.run()

func runTraining(
    graph: CoinGraph,
    startBar: Int,
    endBar: Int?,
    printEvery: Int
) async throws {
    print("\n📊 Training Mode")

    let actualEndBar = endBar ?? min(graph.commonTimestamps.count, 10000)
    print("Training from bar \(startBar) to \(actualEndBar)...")

    // Initialize HRM model
    var model = HRMModel(
        nEdges: graph.edgeState.count,
        hDim: 4,
        zDim: 4,
        yDepth: 200,
        xPixels: 20,
        curvature: 2.0,
        learningRate: 0.001
    )

    model.registerEdges(Array(graph.edgeState.keys))

    var totalLoss = 0.0
    var updateCount = 0

    // Training loop
    for barIdx in startBar..<actualEndBar {
        let (edgeAccels, _, hitPtt, hitStop) = graph.update(barIdx: barIdx)

        if edgeAccels.isEmpty {
            continue
        }

        // Update fisheye for each edge
        for (edge, accel) in edgeAccels {
            model.updateFisheye(edge: edge, value: accel, x: 0, y: 0)
        }

        // Training step (after warmup)
        if barIdx >= 2 {
            for edge in model.edgeNames {
                let fisheyeColumn = model.getFisheye(edge: edge).map { $0[0] }

                if let ptt = hitPtt[edge], let stop = hitStop[edge] {
                    let targetFrac: Double = ptt ? 1.0 : (stop ? 0.0 : 0.5)
                    let targetPtt: Double = ptt ? 1.0 : 0.0
                    let targetStop: Double = stop ? 1.0 : 0.0

                    let loss = trainEdge(
                        model: &model,
                        edge: edge,
                        fisheyeColumn: fisheyeColumn,
                        targetFrac: targetFrac,
                        targetPtt: targetPtt,
                        targetStop: targetStop
                    )

                    if loss != nil {
                        totalLoss += loss!
                        updateCount += 1
                    }
                }
            }
        }

        if barIdx % printEvery == 0 && barIdx > 0 {
            let avgLoss = updateCount > 0 ? totalLoss / Double(updateCount) : 0.0
            print("Bar \(barIdx): avg_loss=\(String(format: "%.6f", avgLoss))")
        }
    }

    let finalAvgLoss = updateCount > 0 ? totalLoss / Double(updateCount) : 0.0
    print("\n✅ Done: avg_loss=\(String(format: "%.6f", finalAvgLoss)), n_updates=\(updateCount)")
}

func runAutoresearch(graph: CoinGraph) async throws {
    print("\n🔬 Autoresearch Mode")

    let phases = 200
    let minPairs = max(4, graph.allPairs.count / 8)
    let maxPairs = graph.allPairs.count
    let totalBars = graph.commonTimestamps.count

    print("Running \(phases) phases of hyperparameter search...")
    print("Pairs: \(minPairs)...\(maxPairs), Bars: \(totalBars)")

    var bestLoss = Double.infinity
    var bestConfig: [String: Any] = [:]

    for phase in 1...phases {
        let progress = Double(phase) / Double(phases)

        // Sample hyperparameters
        let lr = pow(10, Double.random(in: (-4.0)...(-1.5)))
        let yDepth = [100, 200, 300, 400].randomElement() ?? 200
        let xPixels = [10, 15, 20, 30].randomElement() ?? 20
        let curvature = Double.random(in: 0.5...4.0)

        // Sample bag of pairs
        let nPairs = Int.random(in: minPairs...maxPairs)
        let selectedPairs = Array(graph.allPairs.shuffled().prefix(nPairs))

        // Sample window
        let windowBars = Int.random(in: 200...totalBars)
        let maxStart = max(0, totalBars - windowBars)
        let startBar = Int.random(in: 0...maxStart)
        let endBar = startBar + windowBars

        // Create model
        var model = HRMModel(
            nEdges: selectedPairs.count,
            hDim: 4,
            zDim: 4,
            yDepth: yDepth,
            xPixels: xPixels,
            curvature: curvature,
            learningRate: lr
        )

        model.registerEdges(selectedPairs)

        // Train
        var totalLoss = 0.0
        var updateCount = 0

        for barIdx in startBar..<endBar {
            let (edgeAccels, _, hitPtt, hitStop) = graph.update(barIdx: barIdx)

            if edgeAccels.isEmpty { continue }

            if barIdx >= 2 {
                for edge in selectedPairs {
                    let fisheyeColumn = model.getFisheye(edge: edge).map { $0[0] }

                    if let ptt = hitPtt[edge], let stop = hitStop[edge] {
                        let targetFrac: Double = ptt ? 1.0 : (stop ? 0.0 : 0.5)
                        let targetPtt: Double = ptt ? 1.0 : 0.0
                        let targetStop: Double = stop ? 1.0 : 0.0

                        if let loss = trainEdge(
                            model: &model,
                            edge: edge,
                            fisheyeColumn: fisheyeColumn,
                            targetFrac: targetFrac,
                            targetPtt: targetPtt,
                            targetStop: targetStop
                        ) {
                            totalLoss += loss
                            updateCount += 1
                        }
                    }
                }
            }
        }

        let avgLoss = updateCount > 0 ? totalLoss / Double(updateCount) : 999.0

        print("\n=== Phase \(phase) (p=\(String(format: "%.2f", progress))) ===")
        print("  Config: lr=\(String(format: "%.4f", lr)), y_depth=\(yDepth), x_pixels=\(xPixels)")
        print("  curvature=\(String(format: "%.2f", curvature))")
        print("  val_loss=\(String(format: "%.6f", avgLoss)) (Best: \(String(format: "%.6f", bestLoss)))")

        if avgLoss < bestLoss {
            bestLoss = avgLoss
            bestConfig = [
                "lr": lr,
                "yDepth": yDepth,
                "xPixels": xPixels,
                "curvature": curvature,
                "phase": phase
            ]
            print("  ✨ New best!")
        }
    }

    print("\n🏁 Autoresearch complete!")
    print("Best config: \(bestConfig)")
    print("Best loss: \(String(format: "%.6f", bestLoss))")
}

func trainEdge(
    model: inout HRMModel,
    edge: String,
    fisheyeColumn: [Double],
    targetFrac: Double,
    targetPtt: Double,
    targetStop: Double
) -> Double? {
    // This would call into the HRMEdgePredictor
    // For now, return a placeholder loss
    return nil
}
