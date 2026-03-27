//
//  GraphShowdown.swift
//  Autotrade
//
//  Swift port of museum/python/graph_showdown.py.
//  The port keeps the Python control flow intact while using the Swift
//  CoinGraph and HRMModel APIs directly.
//

import Foundation
import AutotradeHRM
import GraphShowdownANE

private let squareCubeSizes = [4, 16, 64, 256]
private let plateauWindow = 100
private let plateauThreshold = 1e-5
private let plateauPatience = 3
private let growthCycle: [GrowthDimension] = [.h, .z, .H, .L]
private let experimentsDB = DuckDB(path: "candles.duckdb")

struct TrainingResult {
    let totalLoss: Double
    let nUpdates: Int
    let earlyStopped: Bool
    let lossHistory: [Double]
    let aneProof: ANEProofSummary?
}

struct ExperimentParams: CustomStringConvertible {
    let lr: Double
    let hDim: Int
    let zDim: Int
    let yDepth: Int
    let xPixels: Int
    let curvature: Double
    let predictionDepth: Int
    let hLayers: Int
    let lLayers: Int

    var description: String {
        "lr=\(String(format: "%.6f", lr)),h=\(hDim),z=\(zDim),y=\(yDepth),x=\(xPixels),curv=\(String(format: "%.2f", curvature)),pred=\(predictionDepth),H=\(hLayers),L=\(lLayers)"
    }
}

struct BagSpec: CustomStringConvertible {
    let nPairs: Int
    let windowBars: Int
    let windowDays: Double
    let startBar: Int

    var description: String {
        "pairs=\(nPairs),bars=\(windowBars),days=\(String(format: "%.1f", windowDays)),start=\(startBar)"
    }
}

struct GraphShowdownArgs {
    let autoresearch: Bool
    let aneTraining: Bool
    let aneRequired: Bool
    let cpuOnly: Bool
    let startBar: Int
    let endBar: Int?
    let printEvery: Int
    let minPartners: Int
    let maxPartners: Int?
    let skipFetch: Bool
    let exchange: String
    let predictionDepth: Int
    let hDim: Int
    let zDim: Int
    let lr: Double
    let yDepth: Int
    let xPixels: Int
    let curvature: Double
}

enum ANETrainingError: LocalizedError, Equatable {
    case initializationFailed(String)
    case backendUnavailable
    case noUpdates
    case noProof(ANEProofSummary)
    case incompatibleFlags(String)

    var errorDescription: String? {
        switch self {
        case .initializationFailed(let message):
            return "ANE required: initialization failed - \(message)"
        case .backendUnavailable:
            return "ANE required: backend entered cpuFallback"
        case .noUpdates:
            return "ANE required: training produced zero updates"
        case .noProof(let proof):
            return "ANE required: proof counters stayed at zero (steps=\(proof.aneStepCount), ane_ms=\(String(format: "%.3f", proof.aneMsTotal)), io_ms=\(String(format: "%.3f", proof.ioMsTotal)))"
        case .incompatibleFlags(let message):
            return "ANE required: \(message)"
        }
    }
}

func formatANEProofSummary(_ proof: ANEProofSummary?) -> String {
    guard let proof else {
        return "ANE proof: unavailable"
    }

    return "ANE proof: backend=\(proof.backendMode.rawValue) steps=\(proof.aneStepCount) ane_ms=\(String(format: "%.3f", proof.aneMsTotal)) io_ms=\(String(format: "%.3f", proof.ioMsTotal))"
}

private func loadCheckpointIfPresent(_ model: inout HRMModel, path: String) {
    do {
        try model.load(path: path)
    } catch let error as HRMCheckpointError {
        if case .fileMissing = error {
            return
        }
        print("Checkpoint load skipped: \(error.localizedDescription)")
    } catch {
        print("Checkpoint load skipped: \(error.localizedDescription)")
    }
}

private func saveCheckpoint(_ model: inout HRMModel, path: String) {
    do {
        try model.save(path: path)
    } catch {
        print("Checkpoint save failed: \(error.localizedDescription)")
    }
}

func createExperimentsTable() {
    do {
        try experimentsDB.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                timestamp TIMESTAMP DEFAULT now(),
                val_bpb DOUBLE,
                params TEXT,
                bag_spec TEXT,
                growth_phase TEXT
            )
            """
        )
    } catch {
        print("Warning: failed to create experiments table: \(error)")
    }
}

func recordExperiment(valBpb: Double, params: ExperimentParams, bagSpec: BagSpec, growthPhase: String) {
    let escapedParams = params.description.replacingOccurrences(of: "'", with: "''")
    let escapedBag = bagSpec.description.replacingOccurrences(of: "'", with: "''")

    do {
        try experimentsDB.execute(
            """
            INSERT INTO experiments (val_bpb, params, bag_spec, growth_phase)
            VALUES (\(valBpb), '\(escapedParams)', '\(escapedBag)', '\(growthPhase)')
            """
        )
    } catch {
        print("Warning: failed to record experiment: \(error)")
    }
}

func isConverged(losses: [Double]) -> Bool {
    guard losses.count >= plateauWindow * plateauPatience else {
        return false
    }

    let n = losses.count
    for patienceIdx in 0..<plateauPatience {
        let start = n - plateauWindow * (plateauPatience - patienceIdx)
        guard start >= 0 else { continue }

        let chunk = Array(losses[start..<n])
        guard chunk.count >= plateauWindow else { continue }

        let recent = chunk.suffix(plateauWindow).reduce(0.0, +) / Double(plateauWindow)
        let older = chunk.prefix(plateauWindow).reduce(0.0, +) / Double(plateauWindow)
        if abs(recent - older) > plateauThreshold {
            return false
        }
    }

    return true
}

func buildPairAdjacency(allPairs: [String]) -> [String: [String]] {
    var adjacency: [String: [String]] = [:]

    for pair in allPairs {
        let parts = pair.split(separator: "-", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { continue }
        for currency in parts {
            adjacency[currency, default: []].append(pair)
        }
    }

    return adjacency
}

func selectRelatedPairs(
    allPairs: [String],
    adj: [String: [String]],
    nPairs: Int,
    rng: inout SeededRandomNumberGenerator
) -> [String] {
    if nPairs >= allPairs.count {
        return allPairs
    }

    let currencies = Array(adj.keys)
    guard !currencies.isEmpty else {
        return Array(allPairs.prefix(nPairs))
    }

    guard let seedCurrency = currencies.randomElement(using: &rng) else {
        return Array(allPairs.prefix(nPairs))
    }

    var selected = Set<String>()
    var orderedSelected: [String] = []
    var frontier = [seedCurrency]
    var visitedCurrencies: Set<String> = [seedCurrency]

    while selected.count < nPairs, !frontier.isEmpty {
        let current = frontier.removeFirst()
        var candidates = (adj[current] ?? []).filter { !selected.contains($0) }
        candidates.shuffle(using: &rng)

        for pair in candidates where selected.count < nPairs {
            guard selected.insert(pair).inserted else { continue }
            orderedSelected.append(pair)

            let parts = pair.split(separator: "-", maxSplits: 1).map(String.init)
            for currency in parts where !visitedCurrencies.contains(currency) {
                visitedCurrencies.insert(currency)
                frontier.append(currency)
            }
        }

        if frontier.isEmpty, selected.count < nPairs {
            let remainingCurrencies = currencies.filter { !visitedCurrencies.contains($0) }
            if let nextSeed = remainingCurrencies.randomElement(using: &rng) {
                visitedCurrencies.insert(nextSeed)
                frontier.append(nextSeed)
            }
        }
    }

    if orderedSelected.count < nPairs {
        for pair in allPairs where !selected.contains(pair) {
            orderedSelected.append(pair)
            if orderedSelected.count == nPairs {
                break
            }
        }
    }

    return orderedSelected
}

func makeTrialGraph(
    fullGraph: CoinGraph,
    selectedPairs: [String],
    startBar: Int,
    endBar: Int
) -> CoinGraph {
    let boundedStart = max(0, startBar)
    let boundedEnd = min(max(boundedStart, endBar), fullGraph.commonTimestamps.count)
    let windowTimestamps = Array(fullGraph.commonTimestamps[boundedStart..<boundedEnd])
    let allowedTimestamps = Set(windowTimestamps)

    let trial = CoinGraph(feeRate: fullGraph.feeRate)
    var edges: [String: [DBCandle]] = [:]
    var edgeState: [String: EdgeState] = [:]
    var nodeState: [String: NodeState] = [:]
    var nodes = Set<String>()

    for pair in selectedPairs {
        guard let candles = fullGraph.edges[pair] else { continue }
        let filtered = candles.filter { allowedTimestamps.contains($0.timestamp) }
        guard !filtered.isEmpty else { continue }

        edges[pair] = filtered
        edgeState[pair] = EdgeState()

        let parts = pair.split(separator: "-", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { continue }
        nodes.insert(parts[0])
        nodes.insert(parts[1])
        nodeState[parts[0]] = NodeState()
        nodeState[parts[1]] = NodeState()
    }

    trial.setEdges(
        edges: edges,
        edgeState: edgeState,
        nodeState: nodeState,
        nodes: nodes,
        allPairs: selectedPairs.filter { edges[$0] != nil },
        commonTimestamps: windowTimestamps
    )
    return trial
}

func runTraining(
    graph: CoinGraph,
    model: inout HRMModel,
    startBar: Int = 0,
    endBar: Int? = nil,
    printEvery: Int = 100,
    lossHistory: [Double] = []
) -> TrainingResult {
    let boundedEnd = min(endBar ?? graph.commonTimestamps.count, graph.commonTimestamps.count)
    var totalLoss = 0.0
    var nUpdates = 0
    var history = lossHistory

    let tsToBar = Dictionary(uniqueKeysWithValues: graph.commonTimestamps.enumerated().map { ($1, $0) })
    var barsWithData = Set<Int>()

    for candles in graph.edges.values {
        for candle in candles {
            if let bar = tsToBar[candle.timestamp] {
                barsWithData.insert(bar)
            }
        }
    }

    let sortedBars = barsWithData.filter { $0 >= startBar && $0 < boundedEnd }.sorted()
    print("Training on \(sortedBars.count) bars with data")

    for (offset, barIdx) in sortedBars.enumerated() {
        guard barIdx < graph.commonTimestamps.count else { break }

        let (edgeAccels, _, hitPtt, hitStop) = graph.update(barIdx: barIdx)
        guard !edgeAccels.isEmpty else { continue }

        if barIdx >= model.predictionDepth {
            _ = model.predict(
                commonTimestamps: graph.commonTimestamps,
                edges: graph.edges,
                barIdx: barIdx
            )
        }

        if barIdx >= model.predictionDepth * 2,
           let loss = model.update(
                commonTimestamps: graph.commonTimestamps,
                edges: graph.edges,
                edgeAccels: edgeAccels,
                barIdx: barIdx,
                hitPtt: hitPtt,
                hitStop: hitStop
           ) {
            totalLoss += loss
            nUpdates += 1
            history.append(loss)
        }

        if offset > 0, offset % printEvery == 0 {
            let avgLoss = nUpdates > 0 ? totalLoss / Double(nUpdates) : 0.0
            print("Bar \(barIdx): avg_loss=\(String(format: "%.6f", avgLoss))")
        }
    }

    return TrainingResult(
        totalLoss: totalLoss,
        nUpdates: nUpdates,
        earlyStopped: false,
        lossHistory: history,
        aneProof: nil
    )
}

func prepareCurvedFeaturesForANE(
    closes: [Double],
    yDepth: Int,
    xPixels: Int,
    curvature: Double
) -> [Double] {
    guard !closes.isEmpty else {
        return Array(repeating: 0.0, count: xPixels)
    }

    let recent = closes.suffix(yDepth)
    let boundaries = fisheyeBoundaries(yDepth: recent.count, xPixels: xPixels, curvature: curvature)
    return fisheyeSample(candles: recent, boundaries: boundaries, xPixels: xPixels)
}

private struct ANEFeatureMux {
    private struct CloseHistory {
        let capacity: Int
        private var storage: [Double]
        private var head: Int
        private var count: Int

        init(capacity: Int) {
            let boundedCapacity = max(1, capacity)
            self.capacity = boundedCapacity
            self.storage = Array(repeating: 0.0, count: boundedCapacity)
            self.head = 0
            self.count = 0
        }

        mutating func append(_ value: Double) {
            guard capacity > 0 else { return }
            if count < capacity {
                storage[count] = value
                count += 1
                return
            }

            storage[head] = value
            head = (head + 1) % capacity
        }

        func orderedValues() -> [Double] {
            guard count > 0 else { return [] }
            if count < capacity {
                return Array(storage.prefix(count))
            }

            var values: [Double] = []
            values.reserveCapacity(count)
            for offset in 0..<count {
                values.append(storage[(head + offset) % capacity])
            }
            return values
        }
    }

    private var cursors: [String: Int]
    private var closeHistory: [String: CloseHistory]

    init(edges: [String: [DBCandle]], maxHistory: Int) {
        self.cursors = Dictionary(uniqueKeysWithValues: edges.keys.map { ($0, 0) })
        self.closeHistory = Dictionary(
            uniqueKeysWithValues: edges.keys.map { ($0, CloseHistory(capacity: maxHistory)) }
        )
    }

    mutating func advanceAll(edges: [String: [DBCandle]], timestamp: Date) {
        for (edge, candles) in edges {
            advance(edge: edge, candles: candles, timestamp: timestamp)
        }
    }

    func closes(edge: String) -> [Double] {
        closeHistory[edge]?.orderedValues() ?? []
    }

    private mutating func advance(edge: String, candles: [DBCandle], timestamp: Date) {
        var cursor = cursors[edge] ?? 0
        while cursor < candles.count, candles[cursor].timestamp < timestamp {
            cursor += 1
        }
        if cursor < candles.count, candles[cursor].timestamp == timestamp {
            var historyForEdge = closeHistory[edge] ?? CloseHistory(capacity: 1)
            historyForEdge.append(candles[cursor].close)
            closeHistory[edge] = historyForEdge
            cursor += 1
        }
        cursors[edge] = cursor
    }
}

func runANETraining(
    graph: CoinGraph,
    config: ANETrainingConfig,
    startBar: Int = 0,
    endBar: Int? = nil,
    printEvery: Int = 100,
    lossHistory: [Double] = [],
    aneRequired: Bool = false,
    makeModel: ((ANETrainingConfig) throws -> ANEModelWrapper)? = nil
) throws -> TrainingResult {
    let boundedEnd = min(endBar ?? graph.commonTimestamps.count, graph.commonTimestamps.count)
    var totalLoss = 0.0
    var nUpdates = 0
    var history = lossHistory

    let aneModel: ANEModelWrapper
    do {
        if let makeModel {
            aneModel = try makeModel(config)
        } else {
            aneModel = try ANEModelWrapper(config: config)
        }
    } catch {
        if aneRequired {
            throw ANETrainingError.initializationFailed(error.localizedDescription)
        }
        print("ANE initialization failed: \(error)")
        return TrainingResult(totalLoss: 0.0, nUpdates: 0, earlyStopped: false, lossHistory: history, aneProof: nil)
    }
    aneModel.registerEdges(Array(graph.edges.keys).sorted())
    if aneModel.backendMode == .cpuFallback {
        if aneRequired {
            throw ANETrainingError.backendUnavailable
        }
        print("ANE runtime unavailable; falling back to the regular HRM training loop.")
        return TrainingResult(totalLoss: 0.0, nUpdates: 0, earlyStopped: false, lossHistory: history, aneProof: aneModel.proofSummary)
    }

    let tsToBar = Dictionary(uniqueKeysWithValues: graph.commonTimestamps.enumerated().map { ($1, $0) })
    var barsWithData = Set<Int>()
    for candles in graph.edges.values {
        for candle in candles {
            if let bar = tsToBar[candle.timestamp] {
                barsWithData.insert(bar)
            }
        }
    }

    let sortedBars = barsWithData.filter { $0 >= startBar && $0 < boundedEnd }.sorted()
    print("ANE training on \(sortedBars.count) bars with data")

    let maxHistory = max(config.seqLen, 2)
    var featureMux = ANEFeatureMux(edges: graph.edges, maxHistory: maxHistory)

    for (offset, barIdx) in sortedBars.enumerated() {
        let timestamp = graph.commonTimestamps[barIdx]
        let (_, _, hitPtt, hitStop) = graph.update(barIdx: barIdx)
        featureMux.advanceAll(edges: graph.edges, timestamp: timestamp)

        for (edge, _) in graph.edges {
            let fisheye = prepareCurvedFeaturesForANE(
                closes: featureMux.closes(edge: edge),
                yDepth: config.seqLen,
                xPixels: config.xPixels,
                curvature: Double(config.curvature)
            )

            let targets = HRMTargets(
                fraction: hitPtt[edge] == true ? 1.0 : (hitStop[edge] == true ? 0.0 : 0.5),
                ptt: hitPtt[edge] == true ? 1.0 : 0.0,
                stop: hitStop[edge] == true ? 1.0 : 0.0
            )

            do {
                let loss = try aneModel.trainStep(
                    edge: edge,
                    fisheye: fisheye,
                    targets: targets
                )
                totalLoss += Double(loss.loss)
                nUpdates += 1
                history.append(Double(loss.loss))
            } catch {
                print("ANE step failed for \(edge): \(error)")
            }
        }

        if offset > 0, offset % printEvery == 0 {
            let avgLoss = nUpdates > 0 ? totalLoss / Double(nUpdates) : 0.0
            print("ANE bar \(barIdx): avg_loss=\(String(format: "%.6f", avgLoss))")
        }
    }

    let proof = aneModel.proofSummary
    if aneRequired && !proof.hasProof {
        throw ANETrainingError.noProof(proof)
    }

    if aneRequired && nUpdates == 0 {
        throw ANETrainingError.noUpdates
    }

    return TrainingResult(
        totalLoss: totalLoss,
        nUpdates: nUpdates,
        earlyStopped: false,
        lossHistory: history,
        aneProof: proof
    )
}

func nextSquareSize(from current: Int) -> Int {
    if let index = squareCubeSizes.firstIndex(of: current), index + 1 < squareCubeSizes.count {
        return squareCubeSizes[index + 1]
    }
    return current
}

func curriculumProgress(hDim: Int, zDim: Int, hLayers: Int, lLayers: Int) -> Double {
    let maxSizeIndex = Double(max(1, squareCubeSizes.count - 1))

    func sizeProgress(_ dim: Int) -> Double {
        if let index = squareCubeSizes.firstIndex(of: dim) {
            return Double(index) / maxSizeIndex
        }

        let clamped = max(dim, squareCubeSizes[0])
        let ratio = log(Double(clamped) / Double(squareCubeSizes[0])) / log(4.0)
        return min(1.0, ratio / maxSizeIndex)
    }

    func layerProgress(_ layers: Int) -> Double {
        min(1.0, Double(max(0, layers - 1)) / 3.0)
    }

    return min(
        1.0,
        (sizeProgress(hDim) + sizeProgress(zDim) + layerProgress(hLayers) + layerProgress(lLayers)) / 4.0
    )
}

func curriculumCeilings(
    totalPairs: Int,
    totalBars: Int,
    progress: Double
) -> (pairUpper: Int, windowUpper: Int, minPairs: Int, minWindowBars: Int) {
    let minPairs = max(4, totalPairs / 8)
    let minWindowBars = max(200, totalBars / 20)
    let pairUpper = minPairs + Int(Double(max(0, totalPairs - minPairs)) * progress)
    let windowUpper = minWindowBars + Int(Double(max(0, totalBars - minWindowBars)) * progress)
    return (
        pairUpper: max(minPairs, pairUpper),
        windowUpper: max(minWindowBars, windowUpper),
        minPairs: minPairs,
        minWindowBars: minWindowBars
    )
}

func validatePowerOf4(_ value: Int, name: String) throws {
    guard value > 0 else {
        throw NSError(domain: "GraphShowdown", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(name) must be positive"])
    }

    var v = value
    while v % 4 == 0 {
        v /= 4
    }

    guard v == 1 else {
        throw NSError(domain: "GraphShowdown", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(name)=\(value) is invalid. Use a power of 4 such as 4, 16, 64, or 256."])
    }
}

func runAutoresearch(graph: CoinGraph, args: GraphShowdownArgs) async throws -> [String: Any]? {
    print("Using HRMModel for hierarchical reasoning")
    createExperimentsTable()

    let adjacency = buildPairAdjacency(allPairs: graph.allPairs)
    let totalBars = graph.commonTimestamps.count
    let totalPairs = graph.allPairs.count
    var rng = SeededRandomNumberGenerator(seed: UInt64(Date().timeIntervalSince1970))

    var bestBpb = Double.infinity
    var growthIdx = 0
    var phase = 0
    var lossHistory: [Double] = []

    var model = HRMModel(
        hDim: args.hDim,
        zDim: args.zDim,
        yDepth: args.yDepth,
        xPixels: args.xPixels,
        curvature: args.curvature,
        learningRate: args.lr,
        predictionDepth: args.predictionDepth,
        hLayers: 2,
        lLayers: 2
    )
    model.registerEdges(Array(graph.edges.keys).sorted())
    loadCheckpointIfPresent(&model, path: "model_weights.pt")

    let openingCeilings = curriculumCeilings(
        totalPairs: totalPairs,
        totalBars: totalBars,
        progress: curriculumProgress(
            hDim: model.hDim,
            zDim: model.zDim,
            hLayers: model.hLayers,
            lLayers: model.lLayers
        )
    )

    print("\nAutoresearch: \(totalPairs) pairs, \(totalBars) bars")
    print("Square Cube: h_dim=\(model.hDim), z_dim=\(model.zDim), H_layers=\(model.hLayers), L_layers=\(model.lLayers)")
    print("Curriculum: pairs [\(openingCeilings.minPairs)..\(totalPairs)], window [\(openingCeilings.minWindowBars)..\(totalBars)]")
    print("Press Ctrl+C to stop.\n")

    while true {
        phase += 1
        let progress = curriculumProgress(
            hDim: model.hDim,
            zDim: model.zDim,
            hLayers: model.hLayers,
            lLayers: model.lLayers
        )
        let ceilings = curriculumCeilings(totalPairs: totalPairs, totalBars: totalBars, progress: progress)

        let requestedPairs = Int.random(in: ceilings.minPairs...ceilings.pairUpper, using: &rng)
        let selectedPairs = selectRelatedPairs(
            allPairs: graph.allPairs,
            adj: adjacency,
            nPairs: requestedPairs,
            rng: &rng
        )

        let actualWindowBars = min(
            totalBars,
            Int.random(in: ceilings.minWindowBars...ceilings.windowUpper, using: &rng)
        )
        let maxStart = max(0, totalBars - actualWindowBars)
        let startBar = maxStart > 0 ? Int.random(in: 0...maxStart, using: &rng) : 0
        let endBar = startBar + actualWindowBars
        let windowDays = round(Double(actualWindowBars) * 5.0 / (60.0 * 24.0) * 10.0) / 10.0

        let trialGraph = makeTrialGraph(
            fullGraph: graph,
            selectedPairs: selectedPairs,
            startBar: startBar,
            endBar: endBar
        )
        guard !trialGraph.edges.isEmpty else {
            print("Phase \(phase): empty graph, skipping")
            continue
        }

        let lr = pow(10.0, Double.random(in: -4.0 ... -1.5, using: &rng))
        let curvature = Double.random(in: 0.5 ... 4.0, using: &rng)
        let predictionDepth = [1, 2, 3, 5, 10].randomElement(using: &rng) ?? 1

        print("\n=== Phase \(phase) (p=\(String(format: "%.2f", progress))) ===")
        print("  Square: h_dim=\(model.hDim), z_dim=\(model.zDim), H_layers=\(model.hLayers), L_layers=\(model.lLayers)")
        print("  Bag: \(trialGraph.edges.count) pairs, \(actualWindowBars) bars (\(windowDays)d)")
        model.learningRate = lr
        model.curvature = curvature
        model.predictionDepth = predictionDepth
        model.registerEdges(Array(trialGraph.edges.keys).sorted())

        let result = runTraining(
            graph: trialGraph,
            model: &model,
            printEvery: 1000,
            lossHistory: lossHistory
        )
        lossHistory = result.lossHistory
        let valBpb = result.nUpdates > 0 ? result.totalLoss / Double(result.nUpdates) : 999.0
        let growthDim = growthCycle[growthIdx]

        print("  val_bpb=\(String(format: "%.6f", valBpb)) (Best: \(String(format: "%.6f", bestBpb)))")

        let params = ExperimentParams(
            lr: lr,
            hDim: model.hDim,
            zDim: model.zDim,
            yDepth: model.yDepth,
            xPixels: model.xPixels,
            curvature: model.curvature,
            predictionDepth: model.predictionDepth,
            hLayers: model.hLayers,
            lLayers: model.lLayers
        )
        let bagSpec = BagSpec(
            nPairs: trialGraph.edges.count,
            windowBars: actualWindowBars,
            windowDays: windowDays,
            startBar: startBar
        )
        recordExperiment(valBpb: valBpb, params: params, bagSpec: bagSpec, growthPhase: growthDim.rawValue)

        if valBpb < bestBpb {
            bestBpb = valBpb
            saveCheckpoint(&model, path: "model_weights.pt")
            print("  --> [NEW BEST] val_bpb: \(String(format: "%.6f", bestBpb))")
        }

        if isConverged(losses: lossHistory) {
            let oldH = model.hDim
            let oldZ = model.zDim
            let oldHLayers = model.hLayers
            let oldLLayers = model.lLayers

            model.grow(dim: growthDim, geometryPolicy: nil)
            lossHistory.removeAll()

            growthIdx = (growthIdx + 1) % growthCycle.count
            print(
                """

                  *** CONVERGED -> GROWTH: \(growthDim.rawValue) \
                [h=\(oldH), z=\(oldZ), H=\(oldHLayers), L=\(oldLLayers)] \
                -> [h=\(model.hDim), z=\(model.zDim), H=\(model.hLayers), L=\(model.lLayers)]
                """
            )
            saveCheckpoint(&model, path: "model_weights_grown.pt")
        } else {
            growthIdx = (growthIdx + 1) % growthCycle.count
        }
    }
}

func parseArgs() -> GraphShowdownArgs {
    let args = Array(CommandLine.arguments.dropFirst())

    func hasFlag(_ name: String) -> Bool {
        args.contains(name)
    }

    func stringValue(_ name: String, default defaultValue: String) -> String {
        guard let index = args.firstIndex(of: name), index + 1 < args.count else {
            return defaultValue
        }
        return args[index + 1]
    }

    func intValue(_ name: String, default defaultValue: Int) -> Int {
        Int(stringValue(name, default: String(defaultValue))) ?? defaultValue
    }

    func doubleValue(_ name: String, default defaultValue: Double) -> Double {
        Double(stringValue(name, default: String(defaultValue))) ?? defaultValue
    }

    let endBarString = stringValue("--end-bar", default: "")
    let maxPartnersString = stringValue("--max-partners", default: "")

    return GraphShowdownArgs(
        autoresearch: hasFlag("--autoresearch"),
        aneTraining: hasFlag("--ane-training"),
        aneRequired: hasFlag("--ane-required"),
        cpuOnly: hasFlag("--cpu-only"),
        startBar: intValue("--start-bar", default: 0),
        endBar: endBarString.isEmpty ? nil : Int(endBarString),
        printEvery: intValue("--print-every", default: 100),
        minPartners: intValue("--min-partners", default: 5),
        maxPartners: maxPartnersString.isEmpty ? nil : Int(maxPartnersString),
        skipFetch: hasFlag("--skip-fetch"),
        exchange: stringValue("--exchange", default: "coinbase"),
        predictionDepth: intValue("--prediction-depth", default: 1),
        hDim: intValue("--h-dim", default: 4),
        zDim: intValue("--z-dim", default: 4),
        lr: doubleValue("--lr", default: 0.001),
        yDepth: intValue("--y-depth", default: 200),
        xPixels: intValue("--x-pixels", default: 20),
        curvature: doubleValue("--curvature", default: 2.0)
    )
}

func runRegularTraining(graph: CoinGraph, args: GraphShowdownArgs, nBars: Int) {
    var model = HRMModel(
        hDim: args.hDim,
        zDim: args.zDim,
        yDepth: args.yDepth,
        xPixels: args.xPixels,
        curvature: args.curvature,
        learningRate: args.lr,
        predictionDepth: args.predictionDepth
    )
    model.registerEdges(Array(graph.edges.keys).sorted())
    loadCheckpointIfPresent(&model, path: "model_weights.pt")

    let endBar = args.endBar ?? min(nBars, 10_000)
    print("Training from bar \(args.startBar) to \(endBar)...")

    let result = runTraining(
        graph: graph,
        model: &model,
        startBar: args.startBar,
        endBar: endBar,
        printEvery: args.printEvery,
        lossHistory: []
    )
    let avgLoss = result.nUpdates > 0 ? result.totalLoss / Double(result.nUpdates) : 0.0
    print("\nDone: avg_loss=\(String(format: "%.6f", avgLoss)), n_updates=\(result.nUpdates)")
    saveCheckpoint(&model, path: "model_weights.pt")
}

func runGraphShowdown() async {
    let args = parseArgs()

    do {
        try validatePowerOf4(args.hDim, name: "h_dim")
        try validatePowerOf4(args.zDim, name: "z_dim")
    } catch {
        print("Error: \(error.localizedDescription)")
        return
    }

    print("Loading coin graph...")
    let graph = CoinGraph(feeRate: 0.001)

    do {
        let nBars = try await graph.load(
            minPartners: args.minPartners,
            exchange: args.exchange,
            skipFetch: args.skipFetch
        )
        print("Loaded \(graph.nodes.count) nodes, \(graph.edges.count) edges, \(nBars) bars")

        guard nBars > 0 else {
            print("No data. Load candles first.")
            return
        }

        if args.autoresearch {
            _ = try await runAutoresearch(graph: graph, args: args)
            return
        }

        if args.aneRequired, args.cpuOnly {
            print("Error: --ane-required cannot be combined with --cpu-only")
            return
        }

        if args.cpuOnly {
            runRegularTraining(graph: graph, args: args, nBars: nBars)
            return
        }

        if args.aneTraining || args.aneRequired {
            let config = ANETrainingConfig(
                dim: args.hDim,
                hidden: args.zDim,
                seqLen: args.yDepth,
                xPixels: args.xPixels,
                curvature: Float(args.curvature),
                heads: 1,
                layers: 2,
                learningRate: Float(args.lr)
            )
            let endBar = args.endBar ?? min(nBars, 10_000)
            print("Attempting ANE training from bar \(args.startBar) to \(endBar)...")
            do {
        let result = try runANETraining(
                graph: graph,
                config: config,
                startBar: args.startBar,
                endBar: endBar,
                    printEvery: args.printEvery,
                    lossHistory: [],
                aneRequired: args.aneRequired
            )

                print(formatANEProofSummary(result.aneProof))
                if let proof = result.aneProof, proof.hasProof, result.nUpdates > 0 {
                    let avgLoss = result.totalLoss / Double(result.nUpdates)
                    print("\nANE done: avg_loss=\(String(format: "%.6f", avgLoss)), n_updates=\(result.nUpdates)")
                    return
                }

                print("ANE path produced no proof. Falling back to regular HRM training...")
            } catch {
                if let aneError = error as? ANETrainingError {
                    switch aneError {
                    case .noProof(let proof):
                        print(formatANEProofSummary(proof))
                    default:
                        break
                    }
                }
                print("Error: \(error.localizedDescription)")
                return
            }
        }

        runRegularTraining(graph: graph, args: args, nBars: nBars)
    } catch {
        print("Error: \(error)")
    }
}

struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
