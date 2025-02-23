// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NNKit",
    platforms: [.macOS(.v14)],
        products: [
        .library(name: "NNKit", targets: ["NNKit"])
    ],
    dependencies: [
        .package(path: "../Tensor")
    ],
    targets: [
        .target(
            name: "NNKit",
            dependencies: [
                "Tensor"
            ],
            path: "Sources/NNKit"
        ),
        .testTarget(
            name: "NNKitTests",
            dependencies: [
                "NNKit"
            ],
            path: "Sources/NNKitTests"
        )
    ]
)
