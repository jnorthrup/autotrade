//
//  AutoresearchTests.swift
//  AutotradeTests
//

import Foundation
import Testing
@testable import Autotrade
@testable import AutotradeHRM

@Suite("Autoresearch Tests")
struct AutoresearchTests {
    @Test("Power sizes remain powers of 4")
    func testSquareCubeSizes() {
        for size in allowedGrowthPowers {
            #expect(isPowerOf4(size))
        }

        for index in 1..<allowedGrowthPowers.count {
            #expect(allowedGrowthPowers[index] > allowedGrowthPowers[index - 1])
        }
    }

    @Test("Plateau detection matches intended sustained flat loss")
    func testPlateauDetection() {
        let converged = Array(repeating: 0.123456, count: 320)
        let stillImproving = (0..<320).map { 1.0 / Double($0 + 1) }

        #expect(isConverged(losses: converged))
        #expect(!isConverged(losses: stillImproving))
    }

    @Test("Curriculum progress tracks model capacity")
    func testCurriculumProgression() {
        let low = curriculumProgress(hDim: 4, zDim: 4, hLayers: 1, lLayers: 1)
        let mid = curriculumProgress(hDim: 16, zDim: 4, hLayers: 2, lLayers: 1)
        let high = curriculumProgress(hDim: 64, zDim: 64, hLayers: 4, lLayers: 4)

        #expect(low < mid)
        #expect(mid < high)
        #expect(high <= 1.0)
    }

    @Test("Bag ceilings grow with capacity and not raw phase")
    func testBagCeilingsTrackCapacity() {
        let early = curriculumCeilings(totalPairs: 120, totalBars: 12_000, progress: 0.0)
        let late = curriculumCeilings(totalPairs: 120, totalBars: 12_000, progress: 1.0)

        #expect(early.pairUpper >= early.minPairs)
        #expect(early.windowUpper >= early.minWindowBars)
        #expect(late.pairUpper > early.pairUpper)
        #expect(late.windowUpper > early.windowUpper)
    }

    @Test("Pair selection stays within requested bag")
    func testPairSelection() {
        let allPairs = ["BTC-USD", "ETH-BTC", "SOL-USD", "DOGE-ETH", "XRP-USD"]
        let adj = buildPairAdjacency(allPairs: allPairs)
        var rng = SeededRandomNumberGenerator(seed: 7)

        let selected = selectRelatedPairs(allPairs: allPairs, adj: adj, nPairs: 3, rng: &rng)

        #expect(selected.count == 3)
        #expect(selected.allSatisfy { allPairs.contains($0) })
    }

    @Test("Power of 4 validation emits exact guidance")
    func testValidatePowerOf4() throws {
        try validatePowerOf4(16, name: "h_dim")
        do {
            try validatePowerOf4(8, name: "h_dim")
            Issue.record("Expected validation failure")
        } catch {
            #expect(error.localizedDescription.contains("power of 4"))
            #expect(error.localizedDescription.contains("4, 16, 64, or 256"))
        }
    }
}
