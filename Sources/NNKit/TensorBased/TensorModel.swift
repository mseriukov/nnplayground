import Tensor

public class TensorModel<Element> where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    private var layers: [any TensorLayer<Element>] = []
    private let optimizer: any Optimizer<Element>

    init(layers: [any TensorLayer<Element>], optimizer: any Optimizer<Element>) {
        self.layers = layers
        self.optimizer = optimizer
    }

    func addLayer(_ layer: any TensorLayer<Element>) {
        layers.append(layer)
    }

    func parameters() -> [TensorParameter<Element>] {
        layers.flatMap { $0.parameters }
    }

    func forward(_ input: Tensor<Element>) -> Tensor<Element> {
        var output = input
        for layer in layers {
            output = layer.forward(output)
        }
        return output
    }

    func backward(_ lossGradient: Tensor<Element>) {
        var gradient = lossGradient
        for layer in layers.reversed() {
            gradient = layer.backward(gradient)
        }
    }

    func train(
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

                // Forward pass
                let outputs = forward(inputs)

                // Compute loss
                let loss = lossFunction.forward(predicted: outputs, actual: targets)
                totalLoss += loss.mean().value

                // Compute loss gradient
                let lossGradient = lossFunction.backward(predicted: outputs, actual: targets)

                // Backward pass
                backward(lossGradient)

                // Apply optimizer step
                optimizer.step()

                batchCount += 1
            }

            print("Epoch \(epoch + 1), Loss: \(totalLoss / Element(batchCount))")
        }
    }
}
