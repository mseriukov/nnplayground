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
        data: (Tensor, Tensor),
        batchSize: Int,
        epochs: Int,
        lossFunction: any LossFunction
    ) {
        for epoch in 0..<epochs {
            let startTimestamp = Date.now.timeIntervalSince1970

            var totalLoss: Tensor.Element = 0.0
            var batchCount = 0

            let size = data.1.shape[0]
            let batches = size / batchSize
            for batch in 0..<batches {
                let start = batch * batchSize

                let currentBatchSize = start + batchSize <= size ? batchSize : size - start

                let inputs = data.0.slice(
                    start: [start, 0],
                    shape: [currentBatchSize, data.0.shape[1]]
                )
                let targets = data.1.slice(
                    start: [start],
                    shape: [currentBatchSize]
                )
                //print("\(batch) \(inputs.shape) \(targets.shape) \(inputs.strides) \(inputs.offset)")
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
            print("Epoch \(epoch + 1), Loss: \(totalLoss / Tensor.Element(batchCount)), Duration: \(duration)")
            print("size: \(Diagnostics.totalSize * 4) Bytes")
        }
    }

    public func verify(
        data: (Tensor, Tensor),
        lossFunction: any LossFunction
    ) -> Tensor.Element {
        let output = forward(data.0)
        let target = data.1
        return lossFunction.forward(predicted: output, actual: target).value
    }
}
