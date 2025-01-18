import Foundation
import Tensor

public class TensorModel {
    private var layers: [any TensorLayer] = []
    private let optimizer: any Optimizer

    public init(layers: [any TensorLayer], optimizer: any Optimizer) {
        self.layers = layers
        self.optimizer = optimizer
    }

    public func parameters() -> [TensorParameter] {
        layers.flatMap { $0.parameters }
    }

    public func forward(_ input: Tensor) -> Tensor {
        var output = input
        for layer in layers {
            output = layer.forward(output)
        }
        return output
    }

    public func backward(_ lossGradient: Tensor) {
        var gradient = lossGradient
        for layer in layers.reversed() {
            gradient = layer.backward(gradient)
        }
    }

    public func train(
        data: [(Tensor, Tensor)],
        batchSize: Int,
        epochs: Int,
        lossFunction: any LossFunction
    ) {
        for epoch in 0..<epochs {
            let startTimestamp = Date.now.timeIntervalSince1970

            var totalLoss: Double = 0.0
            var batchCount = 0

            let chunkedData = data.chunked(into: batchSize)
            for batch in chunkedData {
                let inputs = Tensor.stacked(batch.map { $0.0 })
                let targets = Tensor.stacked(batch.map { $0.1 })

                let outputs = forward(inputs)

                let loss = lossFunction.forward(predicted: outputs, actual: targets)
                totalLoss += loss.value

                parameters().forEach { $0.resetGrad() }

                let lossGradient = lossFunction.backward(predicted: outputs, actual: targets)
                backward(lossGradient)

                optimizer.step()

                batchCount += 1
            }
            let duration = Date.now.timeIntervalSince1970 - startTimestamp
            print("Epoch \(epoch + 1), Loss: \(totalLoss / Double(batchCount)), Duration: \(duration)")
        }
    }

    public func verify(
        data: [(Tensor, Tensor)],
        lossFunction: any LossFunction
    ) -> Double {
        var totalLoss: Double = 0.0
        for example in data {
            let output = forward(example.0)
            let target = example.1
            let loss = lossFunction.forward(predicted: output, actual: target)
            totalLoss += loss.value
        }
        return totalLoss / Double(data.count)
    }
}
