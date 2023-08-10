// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "nnplayground",
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "npPlayground",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                "cnnutils",
                "AlgebraKit"
            ],
            path: "Sources/nnplayground",
            swiftSettings: [.define("ACCELERATE_NEW_LAPACK"), .define("ACCELERATE_LAPACK_ILP64")]
        ),
        .target(
            name: "cnnutils",
            path: "Sources/cnnutils",
            publicHeadersPath: "include"
        ),
        .target(
            name: "AlgebraKit",
            path: "Sources/AlgebraKit",
            swiftSettings: [.define("ACCELERATE_NEW_LAPACK"), .define("ACCELERATE_LAPACK_ILP64")]
        ),
        .testTarget(
            name: "AlgebraKitTests",
            dependencies: [
                "AlgebraKit",
            ],
            path: "Sources/AlgebraKitTests"
        )
    ]
)
