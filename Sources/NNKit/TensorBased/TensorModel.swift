import Tensor

public class TensorModel<Element> where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    private var layers: [any TensorLayer<Element>] = []
    private let optimizer: any Optimizer<Element>

    public init(layers: [any TensorLayer<Element>], optimizer: any Optimizer<Element>) {
        self.layers = layers
        self.optimizer = optimizer
    }

    public func parameters() -> [TensorParameter<Element>] {
        layers.flatMap { $0.parameters }
    }

    public func forward(_ input: Tensor<Element>) -> Tensor<Element> {
        var output = input
        for layer in layers {
            output = layer.forward(output)
        }
        return output
    }

    public func backward(_ lossGradient: Tensor<Element>) {
        var gradient = lossGradient
        for layer in layers.reversed() {
            gradient = layer.backward(gradient)
        }
    }

    public func train(
        data: [(Tensor<Element>, Tensor<Element>)],
        batchSize: Int,
        epochs: Int,
        lossFunction: any LossFunction<Element>
    ) {
        for epoch in 0..<epochs {
            var totalLoss: Element = 0.0
            var batchCount = 0

            for batch in data.chunked(into: batchSize) {
                let inputs = stacked(batch.map { $0.0 })
                let targets = stacked(batch.map { $0.1 })

                let outputs = forward(inputs)

                let loss = lossFunction.forward(predicted: outputs, actual: targets)
                totalLoss += loss.mean().value

                let lossGradient = lossFunction.backward(predicted: outputs, actual: targets)
                backward(lossGradient)

                optimizer.step()

                batchCount += 1
            }

            print("Epoch \(epoch + 1), Loss: \(totalLoss / Element(batchCount))")
        }
    }
}
