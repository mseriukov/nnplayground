import Foundation
import Tensor
import NNKit

class MNISTMLP {
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

    public func train(with dataset: MNISTDataSet) throws {
        model.train(
            data: (try dataset.trainingImages.normalized().reshape([dataset.trainingImages.shape[0], 784]), dataset.trainingLabels),
            batchSize: 128,
            epochs: 10,
            lossFunction: MeanSquaredErrorOneHotActual()
        )

        try verify(with: dataset)
    }

    public func verify(with dataset: MNISTDataSet) throws {
        let testImages = try dataset.testImages.normalized().reshape([dataset.testImages.shape[0], 784])
        let testLabels = dataset.testLabels

        let meanLoss = model.verify(data: (testImages, testLabels), lossFunction: MeanSquaredErrorOneHotActual())
        print("Verification set mean loss: \(meanLoss)")
    }

    private func parseStr(_ input: String) -> [Int] {
        input
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ",")
            .compactMap { Int($0) }

    }



//    private func fromOneHot(_ tensor: Tensor) -> Tensor{
//        precondition(tensor.shape.count == 2)
//        let tensor = tensor.makeContiguous()
//
//        let classesCount = tensor.shape[0]
//        let batchCount = tensor.shape[1]
//
//        var result = Tensor(zeros: [batchCount])
//
//        for batch in 0..<batchCount {
//            let data = tensor.slice(start: [batch, 0], shape: [1, classesCount]).makeContiguous().storage.data
//            guard let hotIndex = data.indices.max(by: { data[$0] < data[$1] }) else  {
//                fatalError("Can't find hot index")
//            }
//            result.assign(Tensor.Element(hotIndex), at: [batch])
//        }
//        return result
//    }
}
