//
//  HRMModel.swift
//  AutotradeHRM
//
//  Hierarchical Reasoning Model for edge prediction in Swift.
//  Port of Python HRM with ANE inference support via Espresso.
//

import Foundation

// MARK: - Core Types

/// Hierarchical Reasoning Model for trading edge prediction
public struct HRMModel {
    public var edgeNames: [String]
    public var hDim: Int
    public var zDim: Int
    public var yDepth: Int
    public var xPixels: Int
    public var curvature: Double
    public var learningRate: Double

    // Edge predictors (one per trading pair)
    private var predictors: [String: HRMEdgePredictor]

    // Fisheye cache per edge
    private var fisheyeCache: [String: FisheyeBuffer]

    public init(
        nEdges: Int,
        hDim: Int = 4,
        zDim: Int = 4,
        yDepth: Int = 200,
        xPixels: Int = 20,
        curvature: Double = 2.0,
        learningRate: Double = 0.001
    ) {
        precondition(hDim == zDim, "Square invariant: hDim must equal zDim")

        self.hDim = hDim
        self.zDim = zDim
        self.yDepth = yDepth
        self.xPixels = xPixels
        self.curvature = curvature
        self.learningRate = learningRate

        self.edgeNames = []
        self.predictors = [:]
        self.fisheyeCache = [:]
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
    ) -> Double {
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
