import Foundation
import Tensor
import NNKit

// MNIST dataset is from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

class TensorMLPTests {
    private var rng: any RandomNumberGenerator = SeedableRandomNumberGenerator(seed: 42)

    var linear1 = TensorLinearLayer(inputDim: 784, outputDim: 500)
    var linear2 = TensorLinearLayer(inputDim: 500, outputDim: 32)
    var linear3 = TensorLinearLayer(inputDim: 32, outputDim: 10)

    let model: TensorModel

    init() {
        model = .init(
           layers: [
               linear1,
               TensorActivationLayer(.relu),
               linear2,
               TensorActivationLayer(.relu),
               linear3,
               TensorActivationLayer(.softmax),
           ],
           optimizer: SGD(
               parameters: [
                   linear1.parameters,
                   linear2.parameters,
                   linear3.parameters
               ].flatMap { $0 },
               learningRate: 0.01
           )
       )

        linear1.parameters.forEach { $0.randomize(&rng) }
        linear2.parameters.forEach { $0.randomize(&rng) }
        linear3.parameters.forEach { $0.randomize(&rng) }
    }

    public func train(inputURL: URL, testURL: URL) throws {
        let reader = FileReader(fileURL: inputURL)
        try reader.open()
        defer { reader.close() }
        // Skip csv column labels.
        _ =  try reader.readLine(maxLength: 16536)

        var examples: [(Tensor, Tensor)] = []

        while let line = try reader.readLine(maxLength: 16536) {
            let nums = parseStr(line)

            let output = nums.first!
            let input = nums.dropFirst()

            examples.append((
                Tensor([input.count], input.map { Tensor.Element($0) }).normalized(), toOneHot(outputLen: 10, n: output)
            ))
        }
        
        model.train(
            data: examples,
            batchSize: 128,
            epochs: 10,
            lossFunction: MeanSquaredError()
        )
        try verify(testURL: testURL)
    }

    public func verify(testURL: URL) throws {
        let reader = FileReader(fileURL: testURL)
        try reader.open()
        defer { reader.close() }
        // Skip csv column labels.
        _ =  try reader.readLine(maxLength: 16536)

        var examples: [(Tensor, Tensor)] = []

        while let line = try reader.readLine(maxLength: 16536) {
            let nums = parseStr(line)

            let output = nums.first!
            let input = nums.dropFirst()

            examples.append((
                Tensor([input.count], input.map { Tensor.Element($0) }).normalized(), toOneHot(outputLen: 10, n: output)
            ))
        }

        let meanLoss = model.verify(data: examples, lossFunction: MeanSquaredError())
        print("Verification set mean loss: \(meanLoss)")
    }

    private func parseStr(_ input: String) -> [Int] {
        input
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ",")
            .compactMap { Int($0) }

    }

    private func toOneHot(outputLen: Int, n: Int) -> Tensor {
        var expected = Tensor.init(zeros: [outputLen])
        expected.assign(1.0, at: [n])
        return expected
    }

    private func fromOneHot(_ tensor: Tensor) -> Tensor{
        precondition(tensor.shape.count == 2)
        let tensor = tensor.makeContiguous()

        let classesCount = tensor.shape[0]
        let batchCount = tensor.shape[1]

        var result = Tensor(zeros: [batchCount])

        for batch in 0..<batchCount {
            let data = tensor.slice(start: [batch, 0], shape: [1, classesCount]).makeContiguous().storage.data
            guard let hotIndex = data.indices.max(by: { data[$0] < data[$1] }) else  {
                fatalError("Can't find hot index")
            }
            result.assign(Tensor.Element(hotIndex), at: [batch])
        }
        return result
    }
}
