import Foundation
import cnnutils
import AlgebraKit
import NNKit

// MNIST dataset is from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

class MLPTests {
    let network: [any Layer] = [
        FullyConnectedLayer(inputSize: 784, outputSize: 500, activation: .sigmoid),
        FullyConnectedLayer(inputSize: 500, outputSize: 32, activation: .sigmoid),
        FullyConnectedLayer(inputSize: 32, outputSize: 10, activation: .sigmoid)
    ]

    func run(url: URL, testURL: URL) throws {
        for i in 0..<10 {
            var matches = 0
            print("epoch: \(i)")
            do {
                let reader = FileReader(fileURL: url)
                try reader.open()
                defer { reader.close() }
                // Discard labels.

                let learningRate: Float = 0.1
                _ =  try reader.readLine(maxLength: 16536)
                var shouldStop = false
                while !shouldStop {
                    try autoreleasepool {
                        guard let line = try reader.readLine(maxLength: 16536) else { shouldStop = true; return }
                        let (input, expected) = parseStr(line)

                        forward(input: input)
                        let output = network.last!.output

                        let errorLocalGrad = loss(output: output, expected: expected)

                        let fwhOut = fromOneHot(output)
                        let fwhExp = fromOneHot(expected)
                        matches += fwhOut == fwhExp ? 1 : 0
                        //print("\(output.storage) - \(fwhExp)")
                        backward(localGradient: errorLocalGrad)
                        network.forEach { $0.updateWeights(eta: learningRate) }
                        network.forEach { $0.resetGrad() }
                    }
                }
            }
            print("matches: \(matches)")
        }
    }

    func forward(input: Matrix) {
        var input: Matrix = input
        for l in network {
            l.forward(input: input)
            input = l.output
        }
    }

    func backward(localGradient: Matrix) {
        var localGradient = localGradient
        for l in network.reversed() {
            localGradient = l.backward(localGradient: localGradient)
        }
    }

    private func parseStr(_ s: String) -> (input: Matrix, expected: Matrix) {
        let nums = s
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ",")
            .compactMap { Float($0) }
        var expected = oneHot(outputLen: 10, n: Int(nums.first!))
        var input = Matrix(rows: 1, cols: 784, data: ContiguousArray(nums.dropFirst()))

        let mean = input.storage.reduce(0.0, +) / Float(input.storage.count)

        let diffsq = input.storage.map({ ($0 - mean) * ($0 - mean) })
        let std_ = diffsq.reduce(0.0, +) / Float(input.storage.count)
        let std = sqrt(std_)

        input = Matrix(as: input, data: ContiguousArray(input.storage.map { ($0 - mean) / std }))

        return (
            input: input,
            expected: expected
        )
    }

    private func loss(output: Matrix, expected: Matrix) -> Matrix {
        output - expected
    }

    private func oneHot(outputLen: Int, n: Int) -> Matrix {
        var expected = Matrix(rows: 1, cols: outputLen, repeating: 0)
        expected[0, n] = 1.0
        return expected
    }

    private func fromOneHot(_ m: Matrix) -> Int {
        return m.storage.indices.max(by: { m.storage[$0] < m.storage[$1] })!
    }
}
