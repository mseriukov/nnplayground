// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "nnplayground",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.1"),
        .package(url: "https://github.com/SnapKit/SnapKit.git", .upToNextMajor(from: "5.0.1"))
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
                "AlgebraKit",
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
            name: "AlgebraKit",
            dependencies: [
                "cnnutils"
            ],
            path: "Sources/AlgebraKit",
            cSettings: [.define("ACCELERATE_NEW_LAPACK")]
        ),
        .testTarget(
            name: "AlgebraKitTests",
            dependencies: [
                "AlgebraKit",
            ],
            path: "Sources/AlgebraKitTests"
        ),
        .target(
            name: "NDArray",
            path: "Sources/NDArray",
            cSettings: [.define("ACCELERATE_NEW_LAPACK")]
        ),
        .testTarget(
            name: "NDArrayTests",
            dependencies: [
                "NDArray",
            ],
            path: "Sources/NDArrayTests"
        ),
        .target(
            name: "NNKit",
            dependencies: [
                "AlgebraKit", "cnnutils"
            ],
            path: "Sources/NNKit"
        ),
        .testTarget(
            name: "NNKitTests",
            dependencies: [
                "NNKit", "AlgebraKit"
            ],
            path: "Sources/NNKitTests"
        )
    ]
)
