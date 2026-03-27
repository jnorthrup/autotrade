//
//  SkillReferenceTests.swift
//  AutotradeTests
//

import Foundation
import Testing

@Suite("Skill Reference Tests")
struct SkillReferenceTests {
    @Test("Repo contains the HRM growth reference skill")
    func testSkillExistsAndUsesExpectedVocabulary() throws {
        let testsDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let repoRoot = testsDir.deletingLastPathComponent().deletingLastPathComponent()
        let skillPath = repoRoot
            .appendingPathComponent(".codex/skills/hrm-growth-reference/SKILL.md")
            .path

        let content = try String(contentsOfFile: skillPath, encoding: .utf8)

        #expect(content.contains("top-left quad is experience"))
        #expect(content.contains("weight forwarded old experience"))
        #expect(content.contains("smooth bricks"))
        #expect(content.contains("bags are stochastic Dykstra picks and stochastic candle spans"))
        #expect(content.contains("ANE changes training economics, not preservation math"))
        #expect(content.contains("backbone"))
        #expect(content.contains("task heads"))
        #expect(content.contains("growth policy"))
    }
}
