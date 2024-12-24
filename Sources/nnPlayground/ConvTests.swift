import Foundation
import AppKit
import cnnutils
import AlgebraKit
import NNKit

// MNIST dataset is from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

class ConvTests {
    let seed: UInt32 = 42
    var imageClosure: ((NSImage?) -> Void)?

    func process(input fileURL: URL, imageClosure: ((NSImage?) -> Void)?) throws {
        self.imageClosure = imageClosure
        let reader = FileReader(fileURL: fileURL)
        try reader.open()
        defer { reader.close() }
        // Discard labels.
        _ =  try reader.readLine(maxLength: 16536)

        var shouldStop = false
        while !shouldStop {
            try autoreleasepool {
                guard let line = try reader.readLine(maxLength: 16536) else { shouldStop = true; return }
                let (input, expected) = parseStr(line)

                var img = input
                img.reshape(size: 28)
                img.scaleToUnitInterval()
                imageClosure?(ImageBuilder.buildImage(from: [img.padded(10)]))
                shouldStop = true
            }
        }
    }

    private func parseStr(_ s: String) -> (input: Matrix, expected: Matrix) {
        let nums = s
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ",")
            .compactMap { Float($0) }
        let expected = toOneHot(outputLen: 10, n: Int(nums.first!))
        var input = Matrix(size: Size(1, 784), data: Array(nums.dropFirst()))
        input.normalize()
        return (
            input: input,
            expected: expected
        )
    }
}
