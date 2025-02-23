import Foundation
import Tensor

struct MNISTDataSet {
    let trainingImages: Tensor
    let trainingLabels: Tensor
    let testImages: Tensor
    let testLabels: Tensor
}

public struct MNISTLoader {
    static func load(from url: URL) throws -> MNISTDataSet {
        .init(
            trainingImages: try IDXLoader.load(from: url.appendingPathComponent("train-images-idx3-ubyte")),
            trainingLabels: try IDXLoader.load(from: url.appendingPathComponent("train-labels-idx1-ubyte")),
            testImages: try IDXLoader.load(from: url.appendingPathComponent("t10k-images-idx3-ubyte")),
            testLabels: try IDXLoader.load(from: url.appendingPathComponent("t10k-labels-idx1-ubyte"))
        )
    }
}
