import Foundation
import cnnutils
import AlgebraKit
import NNKit

// MNIST dataset is from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

class MLPTests {
    var linear1 = LinearLayer(inputSize: 784, outputSize: 500)
    var linear2 = LinearLayer(inputSize: 500, outputSize: 32)
    var linear3 = LinearLayer(inputSize: 32, outputSize: 10)

    lazy var network: [any Layer] = {[
        linear1,
        ActivationLayer(.relu),
        linear2,
        ActivationLayer(.relu),
        linear3,
        ActivationLayer(.softmax),
    ]}()

    lazy var parameters: [Parameter] = {
        network.reduce([], { $0 + $1.parameters })
    }()

    func initializeParameters() {
        // TODO: Float.random doesn't have normal distribution. use LAPACK slarnv() instead.
        linear1.weight.randomize({ Float.random(in: 0...1 * sqrt(2.0 / Float(linear1.weight.value.rows))) })//{ Float.random(in: -0.1...0.1) })
        linear2.weight.randomize({ Float.random(in: 0...1 * sqrt(2.0 / Float(linear2.weight.value.rows))) })//{ Float.random(in: -0.1...0.1) })
        linear3.weight.randomize({ Float.random(in: 0...1 * sqrt(2.0 / Float(linear3.weight.value.rows))) })//{ Float.random(in: -0.1...0.1) })
    }

    // FYI: eta is just a typeable version of η.
    func updateParameters(_ parameters: [Parameter], eta: Float) {
        for parameter in parameters {
            parameter.value -= parameter.grad * eta
            parameter.resetGrad()
        }
    }

    func run(url: URL, testURL: URL) throws {
        initializeParameters()

        for i in 0..<10 {
            var matches = 0
            var total = 0
            print("epoch: \(i)")
            do {
                let reader = FileReader(fileURL: url)
                try reader.open()
                defer { reader.close() }
                // Discard labels.
                _ =  try reader.readLine(maxLength: 16536)

                var shouldStop = false
                while !shouldStop {
                    try autoreleasepool {
                        guard let line = try reader.readLine(maxLength: 16536) else { shouldStop = true; return }
                        let (input, expected) = parseStr(line)

                        let output = forward(input)
                        let errorLocalGrad = loss(output: output, expected: expected)

                        let fwhOut = fromOneHot(output)
                        let fwhExp = fromOneHot(expected)
                        matches += fwhOut == fwhExp ? 1 : 0
                        total += 1
                        backward(errorLocalGrad)
                        updateParameters(parameters, eta: 0.001)
                    }
                }
            }
            print("accuracy: \(Float(matches) / Float(total))")
        }
        print("Verify on test set.")
        var matches = 0
        var total = 0
        do {
            let reader = FileReader(fileURL: testURL)
            try reader.open()
            defer { reader.close() }
            // Discard labels.
            _ =  try reader.readLine(maxLength: 16536)

            var shouldStop = false
            while !shouldStop {
                try autoreleasepool {
                    guard let line = try reader.readLine(maxLength: 16536) else { shouldStop = true; return }
                    let (input, expected) = parseStr(line)

                    let output = forward(input)
                    let fwhOut = fromOneHot(output)
                    let fwhExp = fromOneHot(expected)
                    matches += fwhOut == fwhExp ? 1 : 0
                    total += 1
                }
            }
        }
        print("accuracy: \(Float(matches) / Float(total))")
    }

    func processMinibatch(_ minibatch: [(Matrix, Matrix)]) {
        for (input, expected) in minibatch {
            
        }
    }

    func forward(_ input: Matrix) -> Matrix {
        var input: Matrix = input
        for l in network {
            input = l.forward(input)
        }
        return input
    }

    @discardableResult
    func backward(_ localGradient: Matrix) -> Matrix {
        var localGradient = localGradient
        for l in network.reversed() {
            localGradient = l.backward(localGradient)
        }
        return localGradient
    }

    private func parseStr(_ s: String) -> (input: Matrix, expected: Matrix) {
        let nums = s
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ",")
            .compactMap { Float($0) }
        let expected = toOneHot(outputLen: 10, n: Int(nums.first!))
        var input = Matrix(rows: 1, cols: 784, data: Array(nums.dropFirst()))
        input.normalize()
        return (
            input: input,
            expected: expected
        )
    }

    private func loss(output: Matrix, expected: Matrix) -> Matrix {
        output - expected
    }
}
