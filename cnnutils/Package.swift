// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "cnnutils",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "cnnutils", targets: ["cnnutils"])
    ],
    targets: [
        .target(
            name: "cnnutils",
            path: "Sources/cnnutils",
            publicHeadersPath: "include"
        )
    ]
)
