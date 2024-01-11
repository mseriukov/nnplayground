import Foundation
import AppKit
import cnnutils
import AlgebraKit
import NNKit

// MNIST dataset is from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

class MLPTests {
    let seed: UInt32 = 42

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
        linear1.weight.randomize(.kaiming(inputChannels: linear1.weight.value.rows), seed: seed)
        linear2.weight.randomize(.kaiming(inputChannels: linear2.weight.value.rows), seed: seed)
        linear3.weight.randomize(.kaiming(inputChannels: linear3.weight.value.rows), seed: seed)
    }

    // FYI: eta is just a typeable version of η.
    func updateParameters(_ parameters: [Parameter], eta: Float) {
        for parameter in parameters {
            parameter.value -= parameter.grad * eta
            parameter.resetGrad()
        }
    }

    var imageClosure: ((NSImage?) -> Void)?

    func run(url: URL, testURL: URL, imageClosure: ((NSImage?) -> Void)?) throws {
        self.imageClosure = imageClosure
        initializeParameters()

        for i in 0..<10 {
            //let matrix = Matrix.identity(size: 200)
            let image = ImageBuilder.buildImage(from: linear1.weight.value)
            let surl = url.deletingLastPathComponent().appendingPathComponent("test\(i)", conformingTo: .png)
            try? image?.save(to: surl)
            imageClosure?(image)
            let startTimestamp = Date.now.timeIntervalSince1970
            print("epoch: \(i)")
            let (total, matches) = try process(input: url, onlyInference: false)
            let duration = Date.now.timeIntervalSince1970 - startTimestamp
            print("accuracy: \(Float(matches) / Float(total)) duration: \(duration)")
        }
        print("Verify on test set.")
        let (total, matches) = try process(input: testURL, onlyInference: true)
        print("accuracy: \(Float(matches) / Float(total))")
    }

    func process(input fileURL: URL, onlyInference: Bool) throws -> (Int, Int) {
        var matches = 0
        var total = 0
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

                let output = forward(input)
                let errorLocalGrad = loss(output: output, expected: expected)

                let fwhOut = fromOneHot(output)
                let fwhExp = fromOneHot(expected)
                matches += fwhOut == fwhExp ? 1 : 0
                total += 1
                if !onlyInference {
                    backward(errorLocalGrad)
                    updateParameters(parameters, eta: 0.001)
                }
            }
        }
        return (total, matches)
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
