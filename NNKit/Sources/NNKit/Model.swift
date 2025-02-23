import Foundation
import Tensor

public class Model {
    private var layers: [any Layer] = []
    private let optimizer: any Optimizer

    public init(layers: [any Layer], optimizer: any Optimizer) {
        self.layers = layers
        self.optimizer = optimizer
    }

    public func parameters() -> [Parameter] {
        layers.flatMap { $0.parameters }
    }

    public func forward(_ input: Tensor) throws -> Tensor {
        var output = input
        for layer in layers {
            output = try layer.forward(output)
        }
        return output
    }

    public func backward(_ lossGradient: Tensor) throws {
        var gradient = lossGradient
        for layer in layers.reversed() {
            gradient = try layer.backward(gradient)
        }
    }

    public func train(
        data: (Tensor, Tensor),
        batchSize: Int,
        epochs: Int,
        lossFunction: any LossFunction
    ) throws {
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
                let outputs = try forward(inputs)

                let loss = lossFunction.forward(predicted: outputs, actual: targets)
                totalLoss += loss.value

                parameters().forEach { $0.resetGrad() }

                let lossGradient = lossFunction.backward(predicted: outputs, actual: targets)
                try backward(lossGradient)

                optimizer.step()

                batchCount += 1
            }
            let duration = Date.now.timeIntervalSince1970 - startTimestamp
            print("Epoch \(epoch + 1), Loss: \(totalLoss / Tensor.Element(batchCount)), Duration: \(duration)")            
        }
    }

    public func verify(
        data: (Tensor, Tensor),
        lossFunction: any LossFunction
    ) throws -> Tensor.Element {
        let output = try forward(data.0)
        let target = data.1
        return lossFunction.forward(predicted: output, actual: target).value
    }
}
