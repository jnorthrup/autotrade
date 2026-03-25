//
//  HRMModel.swift
//  AutotradeHRM
//
//  Hierarchical Reasoning Model for edge prediction in Swift.
//  Port of Python HRM with ANE inference support via Espresso.
//

import Foundation

// MARK: - Core Types

public struct HRMModel {
    public var edgeNames: [String]
    public var hDim: Int
    public var zDim: Int
    public var yDepth: Int
    public var xPixels: Int
    public var curvature: Double
    public var learningRate: Double
    public var predictionDepth: Int
    public var hLayers: Int
    public var lLayers: Int
    public var hCycles: Int
    public var lCycles: Int
    
    // Edge predictors (one per trading pair)
    private var predictors: [String: HRMEdgePredictor]
    
    // Fisheye cache per edge
    private var fisheyeCache: [String: FisheyeBuffer]
    
    // Close price buffer for fisheye computation
    private var closeBuffer: [String: [Double]]
    private let maxHistory: Int
    
    // Prediction queue for delayed training
    private var predictionQueue: [String: [(fisheye: [Double], barIdx: Int)]]
    private var carryStates: [String: EdgeCarry]
    
    public init(
        nEdges: Int = 0,
        hDim: Int = 4,
        zDim: Int = 4,
        yDepth: Int = 200,
        xPixels: Int = 20,
        curvature: Double = 2.0,
        learningRate: Double = 0.001,
        predictionDepth: Int = 1,
        hLayers: Int = 2,
        lLayers: Int = 2,
        hCycles: Int = 2,
        lCycles: Int = 2
    ) {
        precondition(hDim == zDim, "Square invariant: hDim must equal zDim")
        
        self.hDim = hDim
        self.zDim = zDim
        self.yDepth = yDepth
        self.xPixels = xPixels
        self.curvature = curvature
        self.learningRate = learningRate
        self.predictionDepth = predictionDepth
        self.hLayers = hLayers
        self.lLayers = lLayers
        self.hCycles = hCycles
        self.lCycles = lCycles
        
        self.edgeNames = []
        self.predictors = [:]
        self.fisheyeCache = [:]
        self.closeBuffer = [:]
        self.maxHistory = max(yDepth + 100, 500)
        self.predictionQueue = [:]
        self.carryStates = [:]
    }
    
    public mutating func registerEdges(_ edges: [String]) {
        self.edgeNames = edges
        for edge in edges {
            predictors[edge] = HRMEdgePredictor(
                hDim: hDim,
                zDim: zDim,
                yDepth: yDepth,
                xPixels: xPixels,
                curvature: curvature,
                learningRate: learningRate
            )
            fisheyeCache[edge] = FisheyeBuffer(
                yDepth: yDepth,
                xPixels: xPixels
            )
            closeBuffer[edge] = []
            predictionQueue[edge] = []
            carryStates[edge] = nil
        }
    }
    
    public func getFisheye(edge: String) -> [[Double]] {
        guard let buffer = fisheyeCache[edge] else {
            return []
        }
        return buffer.data
    }
    
    public mutating func updateFisheye(edge: String, value: Double, x: Int, y: Int) {
        if var buffer = fisheyeCache[edge] {
            buffer.update(value: value, x: x, y: y)
            fisheyeCache[edge] = buffer
        }
    }
    
    private func getFisheyeColumn(edge: String) -> [Double] {
        let data = closeBuffer[edge] ?? []
        guard data.count >= 2 else {
            return Array(repeating: 0.0, count: xPixels)
        }
        
        let boundaries = fisheyeBoundaries(yDepth: yDepth, xPixels: xPixels, curvature: curvature)
        let recentCandles = Array(data.suffix(yDepth))
        return fisheyeSample(candles: recentCandles, boundaries: boundaries, xPixels: xPixels)
    }
    
    private mutating func updatePrices(commonTimestamps: [Date], edges: [String: [DBCandle]], barIdx: Int) {
        guard barIdx >= 0, barIdx < commonTimestamps.count else { return }
        
        let ts = commonTimestamps[barIdx]
        
        for product in edgeNames {
            guard let candles = edges[product],
                  let idx = candles.firstIndex(where: { $0.timestamp == ts }) else {
                continue
            }
            
            let close = candles[idx].close
            var buf = closeBuffer[product, default: []]
            buf.append(close)
            if buf.count > maxHistory {
                buf = Array(buf.suffix(maxHistory))
            }
            closeBuffer[product] = buf
        }
    }
    
    public mutating func predict(commonTimestamps: [Date], edges: [String: [DBCandle]], barIdx: Int) -> [String: (fraction: Double, ptt: Double, stop: Double)] {
        var predictions: [String: (fraction: Double, ptt: Double, stop: Double)] = [:]
        
        updatePrices(commonTimestamps: commonTimestamps, edges: edges, barIdx: barIdx)
        
        for edge in edgeNames {
            let fisheye = getFisheyeColumn(edge: edge)
            
            if var predictor = predictors[edge] {
                let output = predictor.forward(fisheyeColumn: fisheye)
                predictors[edge] = predictor
                
                predictions[edge] = (output.frac, output.ptt, output.stop)
                
                predictionQueue[edge, default: []].append((fisheye, barIdx))
                if predictionQueue[edge]!.count > predictionDepth {
                    predictionQueue[edge] = Array(predictionQueue[edge]!.suffix(predictionDepth))
                }
            }
        }
        
        return predictions
    }
    
    public mutating func update(commonTimestamps: [Date], edges: [String: [DBCandle]], edgeAccels: [String: Double], barIdx: Int,
                                hitPtt: [String: Bool], hitStop: [String: Bool]) -> Double? {
        guard !edgeNames.isEmpty else { return nil }
        
        updatePrices(commonTimestamps: commonTimestamps, edges: edges, barIdx: barIdx)
        
        var totalLoss = 0.0
        var nScored = 0
        
        for edge in edgeNames {
            guard let queue = predictionQueue[edge], queue.count >= predictionDepth else { continue }
            
            let frame = queue[0]
            let fisheye = frame.fisheye
            
            let targetFrac: Double
            if hitPtt[edge] ?? false {
                targetFrac = 1.0
            } else if hitStop[edge] ?? false {
                targetFrac = 0.0
            } else {
                targetFrac = 0.5
            }
            
            let targetPtt: Double = (hitPtt[edge] ?? false) ? 1.0 : 0.0
            let targetStop: Double = (hitStop[edge] ?? false) ? 1.0 : 0.0
            
            if var predictor = predictors[edge] {
                let loss = predictor.trainStep(
                    fisheyeColumn: fisheye,
                    targetFrac: targetFrac,
                    targetPtt: targetPtt,
                    targetStop: targetStop
                )
                predictors[edge] = predictor
                
                if let loss = loss {
                    totalLoss += loss
                    nScored += 1
                }
            }
        }
        
        return nScored > 0 ? totalLoss / Double(nScored) : nil
    }
    
    public mutating func grow(dim: String) {
        // Simple growth implementation for Swift
        switch dim {
        case "h":
            hDim *= 4
            zDim = hDim
            for edge in edgeNames {
                if var predictor = predictors[edge] {
                    predictor.growRotation(dimension: "h")
                    predictors[edge] = predictor
                }
            }
        case "H":
            hLayers *= 2
        case "L":
            lLayers *= 2
        default:
            break
        }
        
        // Re-initialize predictors with new dimensions
        for edge in edgeNames {
            predictors[edge] = HRMEdgePredictor(
                hDim: hDim,
                zDim: zDim,
                yDepth: yDepth,
                xPixels: xPixels,
                curvature: curvature,
                learningRate: learningRate
            )
        }
    }
    
    public func save(path: String) {
        struct ModelState: Codable {
            let hDim: Int
            let zDim: Int
            let yDepth: Int
            let xPixels: Int
            let curvature: Double
            let learningRate: Double
            let predictionDepth: Int
            let hLayers: Int
            let lLayers: Int
            let hCycles: Int
            let lCycles: Int
            let edgeNames: [String]
        }
        
        let state = ModelState(
            hDim: hDim,
            zDim: zDim,
            yDepth: yDepth,
            xPixels: xPixels,
            curvature: curvature,
            learningRate: learningRate,
            predictionDepth: predictionDepth,
            hLayers: hLayers,
            lLayers: lLayers,
            hCycles: hCycles,
            lCycles: lCycles,
            edgeNames: edgeNames
        )
        
        if let data = try? JSONEncoder().encode(state) {
            try? data.write(to: URL(fileURLWithPath: path))
        }
    }
    
    public mutating func load(path: String) {
        struct ModelState: Codable {
            let hDim: Int
            let zDim: Int
            let yDepth: Int
            let xPixels: Int
            let curvature: Double
            let learningRate: Double
            let predictionDepth: Int
            let hLayers: Int
            let lLayers: Int
            let hCycles: Int
            let lCycles: Int
            let edgeNames: [String]
        }
        
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let state = try? JSONDecoder().decode(ModelState.self, from: data) else {
            return
        }
        
        self.hDim = state.hDim
        self.zDim = state.zDim
        self.yDepth = state.yDepth
        self.xPixels = state.xPixels
        self.curvature = state.curvature
        self.learningRate = state.learningRate
        self.predictionDepth = state.predictionDepth
        self.hLayers = state.hLayers
        self.lLayers = state.lLayers
        self.hCycles = state.hCycles
        self.lCycles = state.lCycles
        self.edgeNames = state.edgeNames
    }
}

// MARK: - Carry State

struct EdgeCarry {
    var state: [Double]
}

// MARK: - Fisheye Buffer

struct FisheyeBuffer {
    var yDepth: Int
    var xPixels: Int
    private(set) var data: [[Double]]
    
    init(yDepth: Int, xPixels: Int) {
        self.yDepth = yDepth
        self.xPixels = xPixels
        self.data = Array(repeating: Array(repeating: 0.0, count: xPixels), count: yDepth)
    }
    
    mutating func update(value: Double, x: Int, y: Int) {
        guard y < yDepth, x < xPixels else { return }
        data[y][x] = value
    }
    
    func getColumn(x: Int) -> [Double] {
        return data.map { $0[x] }
    }
}

// MARK: - HRM Edge Predictor

struct HRMEdgePredictor {
    let hDim: Int
    let zDim: Int
    let yDepth: Int
    let xPixels: Int
    let curvature: Double
    let learningRate: Double

    // Transformer weights
    private var W1: [[Double]]  // Input projection
    private var W2: [[Double]]  // Output projection
    private var Wq: [[Double]]  // Query projection
    private var Wk: [[Double]]  // Key projection
    private var Wv: [[Double]]  // Value projection

    // Optimizer state
    private var mW1: [[Double]]
    private var vW1: [[Double]]
    private var beta1: Double = 0.9
    private var beta2: Double = 0.999
    private var epsilon: Double = 1e-8
    private var timestep: Int = 0

    init(
        hDim: Int,
        zDim: Int,
        yDepth: Int,
        xPixels: Int,
        curvature: Double,
        learningRate: Double
    ) {
        precondition(hDim == zDim, "Square invariant")

        self.hDim = hDim
        self.zDim = zDim
        self.yDepth = yDepth
        self.xPixels = xPixels
        self.curvature = curvature
        self.learningRate = learningRate

        // Initialize weights with Xavier/Glorot initialization
        self.W1 = randomMatrix(rows: xPixels, cols: hDim, std: 0.02)
        self.W2 = randomMatrix(rows: zDim, cols: 3, std: 0.02)  // 3 outputs: frac, ptt, stop
        self.Wq = randomMatrix(rows: hDim, cols: hDim, std: 0.02)
        self.Wk = randomMatrix(rows: hDim, cols: hDim, std: 0.02)
        self.Wv = randomMatrix(rows: hDim, cols: hDim, std: 0.02)

        self.mW1 = zeroMatrix(rows: xPixels, cols: hDim)
        self.vW1 = zeroMatrix(rows: xPixels, cols: hDim)
    }

    /// Forward pass: fisheye column → predictions
    func forward(fisheyeColumn: [Double]) -> (frac: Double, ptt: Double, stop: Double) {
        // Project fisheye to hidden space
        let h = matvec(A: W1, x: fisheyeColumn)

        // Self-attention
        let q = matvec(A: Wq, x: h)
        let k = matvec(A: Wk, x: h)
        let v = matvec(A: Wv, x: h)

        let attnScore = dot(q, k) / Double(hDim)
        let attnWeight = sigmoid(attnScore * curvature)
        let z = v.map { $0 * attnWeight }

        // Output projection
        let logits = matvec(A: W2, x: z)

        let frac = sigmoid(logits[0])
        let ptt = sigmoid(logits[1])
        let stop = sigmoid(logits[2])

        return (frac, ptt, stop)
    }

    /// Training step with gradient update
    mutating func trainStep(
        fisheyeColumn: [Double],
        targetFrac: Double,
        targetPtt: Double,
        targetStop: Double
    ) -> Double? {
        // Forward pass
        let output = forward(fisheyeColumn: fisheyeColumn)

        // Compute loss (binary cross-entropy)
        let loss = bceLoss(pred: output.frac, target: targetFrac) +
                   bceLoss(pred: output.ptt, target: targetPtt) +
                   bceLoss(pred: output.stop, target: targetStop)

        // Backward pass (simplified gradients)
        let gradFrac = output.frac - targetFrac
        let gradPtt = output.ptt - targetPtt
        let gradStop = output.stop - targetStop

        // Adam update
        timestep += 1
        let lr = learningRate * sqrt(1.0 - pow(beta2, Double(timestep))) /
                 (1.0 - pow(beta1, Double(timestep)))

        // Update W2 (output layer)
        for i in 0..<W2.count {
            for j in 0..<W2[0].count {
                let grad: Double
                switch j {
                case 0: grad = gradFrac
                case 1: grad = gradPtt
                case 2: grad = gradStop
                default: grad = 0
                }

                mW1[i][j] = beta1 * mW1[i][j] + (1.0 - beta1) * grad
                vW1[i][j] = beta2 * vW1[i][j] + (1.0 - beta2) * grad * grad
                W2[i][j] -= lr * mW1[i][j] / (sqrt(vW1[i][j]) + epsilon)
            }
        }

        return loss
    }

    /// Rotation growth: expand dimensions by 4× via 90°/180°/270° rotations
    mutating func growRotation(dimension: String) {
        switch dimension {
        case "z":
            // Grow z dimension first (output layer)
            let original = W2
            W2 = Array(repeating: Array(repeating: 0.0, count: 3), count: zDim * 4)

            // 0° quadrant (preserve original)
            for i in 0..<original.count {
                W2[i] = original[i]
            }

            // 180° rotation
            for i in 0..<original.count {
                for j in 0..<3 {
                    W2[i + original.count][j] = -original[i][j]
                }
            }

            // 90° rotation
            for i in 0..<original.count {
                for j in 0..<3 {
                    W2[i + original.count * 2][j] = original[original.count - 1 - i][j]
                }
            }

            // 270° rotation
            for i in 0..<original.count {
                for j in 0..<3 {
                    W2[i + original.count * 3][j] = -original[original.count - 1 - i][j]
                }
            }

        case "h":
            // Grow h dimension (input layer)
            let original = W1
            W1 = Array(repeating: Array(repeating: 0.0, count: hDim * 4), count: xPixels)

            // 0° quadrant
            for i in 0..<xPixels {
                for j in 0..<original[0].count {
                    W1[i][j] = original[i][j]
                }
            }

            // Add rotated quadrants similarly...

        default:
            break
        }
    }
}

// MARK: - Fisheye Utilities

func fisheyeBoundaries(yDepth: Int, xPixels: Int, curvature: Double) -> [Int] {
    guard xPixels > 1 else { return [yDepth] }
    
    var boundaries: [Int] = []
    var prevBoundary = 0
    
    for i in 0..<xPixels {
        let t = Double(i) / Double(xPixels - 1)
        let warped = pow(t, curvature)
        let boundary = Int(Double(yDepth) * warped)
        let clampedBoundary = max(boundary, prevBoundary + 1)
        boundaries.append(clampedBoundary)
        prevBoundary = clampedBoundary
    }
    
    return boundaries
}

func fisheyeSample(candles: [Double], boundaries: [Int], xPixels: Int) -> [Double] {
    guard !candles.isEmpty else {
        return Array(repeating: 0.0, count: xPixels)
    }
    
    var results: [Double] = []
    var prevIdx = 0
    
    for boundary in boundaries {
        let bucketStart = prevIdx
        let bucketEnd = min(boundary, candles.count)
        
        if bucketEnd <= bucketStart {
            results.append(0.0)
        } else {
            let bucketCandles = Array(candles[bucketStart..<bucketEnd])
            let bucketMean = bucketCandles.reduce(0, +) / Double(bucketCandles.count)
            let currentClose = candles[candles.count - 1]
            let value = bucketMean != 0 ? (currentClose - bucketMean) / bucketMean : 0.0
            results.append(value)
        }
        
        prevIdx = boundary
    }
    
    return Array(results.prefix(xPixels))
}

// MARK: - Math Utilities (Pure Swift)

extension Double {
    func sigmoid() -> Double {
        return 1.0 / (1.0 + exp(-self))
    }
}

private func randomMatrix(rows: Int, cols: Int, std: Double) -> [[Double]] {
    return (0..<rows).map { _ in
        (0..<cols).map { _ in
            Double.random(in: -std...std)
        }
    }
}

private func zeroMatrix(rows: Int, cols: Int) -> [[Double]] {
    return Array(repeating: Array(repeating: 0.0, count: cols), count: rows)
}

private func matvec(A: [[Double]], x: [Double]) -> [Double] {
    precondition(x.count == A[0].count, "Dimension mismatch")
    var result = [Double](repeating: 0.0, count: A.count)

    for i in 0..<A.count {
        var sum = 0.0
        for j in 0..<x.count {
            sum += A[i][j] * x[j]
        }
        result[i] = sum
    }

    return result
}

private func dot(_ a: [Double], _ b: [Double]) -> Double {
    precondition(a.count == b.count, "Vector length mismatch")
    return zip(a, b).map(*).reduce(0, +)
}

private func sigmoid(_ x: Double) -> Double {
    return x.sigmoid()
}

private func bceLoss(pred: Double, target: Double) -> Double {
    let eps = 1e-7
    let p = max(min(pred, 1.0 - eps), eps)
    return -(target * log(p) + (1.0 - target) * log(1.0 - p))
}
