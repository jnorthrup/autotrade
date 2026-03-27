// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Autotrade",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "Autotrade", targets: ["Autotrade"]),
        .library(name: "AutotradeHRM", targets: ["AutotradeHRM"]),
        .library(name: "GraphShowdownANE", targets: ["GraphShowdownANE"])
    ],
    dependencies: [
        .package(url: "https://github.com/duckdb/duckdb-swift", .upToNextMajor(from: .init(1, 0, 0))),
        .package(path: "museum/ane/Espresso")
    ],
    targets: [
        .executableTarget(
            name: "Autotrade",
            dependencies: ["AutotradeHRM", "GraphShowdownANE"],
            path: "Sources/Autotrade",
            exclude: ["GRAPHDOWDOWN_SWIFT.md"]
        ),
        .target(
            name: "AutotradeHRM",
            dependencies: [
                .product(name: "DuckDB", package: "duckdb-swift")
            ],
            path: "Sources/AutotradeHRM"
        ),
        .target(
            name: "GraphShowdownANE",
            dependencies: [
                "AutotradeHRM",
                .product(name: "Espresso", package: "Espresso")
            ],
            path: "Sources/GraphShowdownANE",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "AutotradeTests",
            dependencies: ["Autotrade", "AutotradeHRM", "GraphShowdownANE"],
            path: "Tests/AutotradeTests"
        )
    ]
)
