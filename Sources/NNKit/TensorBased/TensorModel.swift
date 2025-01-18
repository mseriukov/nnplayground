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
            var totalLoss: Double = 0.0
            var batchCount = 0

            let chunkedData = data.chunked(into: batchSize)
            for batch in chunkedData {
                let inputs = Tensor.stacked(batch.map { $0.0 })
                let targets = Tensor.stacked(batch.map { $0.1 })

                let outputs = forward(inputs)

                let loss = lossFunction.forward(predicted: outputs, actual: targets)
                totalLoss += loss.mean().value

                let lossGradient = lossFunction.backward(predicted: outputs, actual: targets)
                backward(lossGradient)

                optimizer.step()

                batchCount += 1
                print("Epoch \(epoch + 1), batch: \(batchCount) / \(chunkedData.count)")
            }

            print("Epoch \(epoch + 1), Loss: \(totalLoss / Double(batchCount))")
        }
    }
}
