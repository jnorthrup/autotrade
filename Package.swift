// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Autotrade",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "Autotrade", targets: ["Autotrade"]),
        .library(name: "AutotradeHRM", targets: ["AutotradeHRM"])
    ],
    dependencies: [
        .package(url: "https://github.com/duckdb/duckdb-swift", .upToNextMajor(from: .init(1, 0, 0)))
    ],
    targets: [
        .executableTarget(
            name: "Autotrade",
            dependencies: ["AutotradeHRM"],
            path: "Sources/Autotrade"
        ),
        .target(
            name: "AutotradeHRM",
            dependencies: [
                .product(name: "DuckDB", package: "duckdb-swift")
            ],
            path: "Sources/AutotradeHRM"
        ),
        .testTarget(
            name: "AutotradeTests",
            dependencies: ["AutotradeHRM"],
            path: "Tests/AutotradeTests"
        )
    ]
)