import Foundation
import Tensor
import NNKit

// MNIST dataset is from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

class TensorMLPTests {
    typealias Element = Double
    private var rng: any RandomNumberGenerator = SeedableRandomNumberGenerator(seed: 42)

    var linear1 = TensorLinearLayer<Element>(inputDim: 784, outputDim: 500)
    var linear2 = TensorLinearLayer<Element>(inputDim: 500, outputDim: 32)
    var linear3 = TensorLinearLayer<Element>(inputDim: 32, outputDim: 10)

    let model: TensorModel<Element>

    init() {
        model = .init(
           layers: [
               linear1,
               TensorActivationLayer<Element>(.relu),
               linear2,
               TensorActivationLayer<Element>(.relu),
               linear3,
               TensorActivationLayer<Element>(.softmax),
           ],
           optimizer: SGD<Element>(
               parameters: [
                   linear1.parameters,
                   linear2.parameters,
                   linear3.parameters
               ].flatMap { $0 },
               learningRate: 0.1
           )
       )

        linear1.parameters.forEach { $0.randomize(&rng) }
        linear2.parameters.forEach { $0.randomize(&rng) }
        linear3.parameters.forEach { $0.randomize(&rng) }
    }

    public func train(inputURL: URL) throws {
        let reader = FileReader(fileURL: inputURL)
        try reader.open()
        defer { reader.close() }
        // Skip csv column labels.
        _ =  try reader.readLine(maxLength: 16536)

        var examples: [(Tensor<Element>, Tensor<Element>)] = []

        while let line = try reader.readLine(maxLength: 16536) {
            let nums = parseStr(line)

            let output = nums.first!
            let input = nums.dropFirst()

            examples.append((
                Tensor<Element>([input.count], input.map { Element($0) }), Tensor<Element>(shape: [1], value: Element(output))
            ))
        }

        model.train(
            data: examples,
            batchSize: 16,
            epochs: 1,
            lossFunction: MeanSquaredError()
        )
    }

    private func parseStr(_ input: String) -> [Int] {
        input
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ",")
            .compactMap { Int($0) }

    }

    private func toOneHot(outputLen: Int, n: Int) -> Tensor<Element> {
        var expected = Tensor<Element>.init(zeros: [1, outputLen])
        expected.assign(1.0, at: [0, n])
        return expected
    }

    private func fromOneHot(_ tensor: Tensor<Element>) -> Tensor<Element> {
        precondition(tensor.shape.count == 2)
        let tensor = tensor.makeContiguous()

        let classesCount = tensor.shape[0]
        let batchCount = tensor.shape[1]

        var result = Tensor<Element>(zeros: [batchCount])

        for batch in 0..<batchCount {
            let data = tensor.slice(start: [batch, 0], shape: [1, classesCount]).makeContiguous().storage.data
            guard let hotIndex = data.indices.max(by: { data[$0] < data[$1] }) else  {
                fatalError("Can't find hot index")
            }
            result.assign(Element(hotIndex), at: [batch])
        }
        return result
    }
}
