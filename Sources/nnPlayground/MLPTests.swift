import Foundation
import cnnutils
import AlgebraKit

// MNIST dataset is from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

class MLPTests {
    let network: [any Layer] = [
        FullyConnectedLayer(inputSize: 784, outputSize: 500, activation: .relu),
        FullyConnectedLayer(inputSize: 500, outputSize: 10, activation: .relu)
    ]

    func run(url: URL, testURL: URL) throws {
        do {
            let reader = FileReader(fileURL: url)
            try reader.open()
            defer { reader.close() }
            // Discard labels.

            let learningRate: Float = 0.000000001
            _ =  try reader.readLine(maxLength: 16536)
            while true {
                guard let line = try reader.readLine(maxLength: 16536) else { break }
                let (input, expected) = parseStr(line)

                forward(input: input)
                let errorLocalGrad = loss(output: network.last!.output, expected: expected)
                print("\(fromOneHot(network.last!.output) ) - \(fromOneHot(expected))")
                backward(localGradient: errorLocalGrad)
                network.forEach { $0.updateWeights(eta: learningRate) }
                network.forEach { $0.resetGrad() }
            }
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

        // TODO: normalize with mean/STD
        input = input / 255.0
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
