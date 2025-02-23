// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Tensor",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "Tensor", targets: ["Tensor"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "Tensor",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            path: "Sources/Tensor",
            cSettings: [.define("ACCELERATE_NEW_LAPACK")]
        ),
        .testTarget(
            name: "TensorTests",
            dependencies: [
                "Tensor"
            ],
            path: "Sources/TensorTests"
        )
    ]
)
