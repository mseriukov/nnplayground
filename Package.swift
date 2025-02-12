// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "nnplayground",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/SnapKit/SnapKit.git", from: "5.7.1")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "nnplayground",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "SnapKit", package: "SnapKit"),
                "cnnutils",
                "Tensor",
                "NNKit"
            ],
            path: "Sources/nnplayground",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "cats_gs",
                    "-Xlinker", "Resources/cats_gs.png"
                ]),
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "cats",
                    "-Xlinker", "Resources/cats.png"
                ]),
            ]
        ),
        .target(
            name: "cnnutils",
            path: "Sources/cnnutils",
            publicHeadersPath: "include"
        ),
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
        ),
        .target(
            name: "NNKit",
            dependencies: [
                "cnnutils", "Tensor"
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
