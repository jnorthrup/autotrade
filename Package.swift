// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Autotrade",
    platforms: [.macOS(.v15)],
    dependencies: [
        // Add Espresso as a local dependency
        // .Package(url: "https://github.com/jnorthrup/Espresso", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "Autotrade",
            dependencies: ["AutotradeHRM"]
        ),
        .target(
            name: "AutotradeHRM",
            dependencies: [],
            path: "Sources/AutotradeHRM"
        ),
        .testTarget(
            name: "AutotradeTests",
            dependencies: ["Autotrade"]
        ),
        .executableTarget(
            name: "autotrade",
            dependencies: ["Autotrade"],
            path: "Sources/autotrade"
        ),
    ]
)
