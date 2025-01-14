import Tensor

public class TensorLinearLayer<Element> where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    var weights: TensorParameter<Element> // Shape: [output, input]
    var bias: TensorParameter<Element>? // Shape: [output]

    // Cache for forward input
    private var cachedInput: Tensor<Element>?
    private var randomGenerator: any RandomNumberGenerator = SystemRandomNumberGenerator()

    init(inputDim: Int, outputDim: Int, includeBias: Bool = true) {
        weights = TensorParameter(tensor: Tensor.random(
            shape: [outputDim, inputDim],
            distribution: .kaiming(channels: inputDim),
            generator: &randomGenerator
        ))
        bias = includeBias ? TensorParameter(tensor: Tensor.random(
            shape: [outputDim],
            distribution: .kaiming(channels: inputDim),
            generator: &randomGenerator
        )) : nil
    }

    func forward(input: Tensor<Element>) -> Tensor<Element> {
        var input = input
        // Ensure we have batch dimension.
        if input.rank == 1 {
            input.unsqueeze(axis: 0)
        }
        precondition(input.shape.last == weights.value.shape.last, "Input dimension must match weight's input_dim.")
        var output = input.matmul(weights.value.transposed())
        cachedInput = input
        if let b = bias?.value.broadcastTo(output.shape) {
            output.add(b)
        }
        return output
    }

    func backward(localGradient: Tensor<Element>) -> Tensor<Element> {
        guard let input = self.cachedInput else {
            fatalError("No cached input. Did you forget to perform a forward pass?")
        }

        let gradInput = localGradient.matmul(weights.value.transposed())

        weights.gradient?.add(input.transposed().matmul(localGradient))
        bias?.gradient?.add(localGradient.sum(alongAxis: 0))

        cachedInput = nil
        return gradInput
    }
}
