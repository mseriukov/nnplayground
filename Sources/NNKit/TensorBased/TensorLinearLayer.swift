import Tensor

public class TensorLinearLayer<Element: BinaryFloatingPoint> {
    var weights: Tensor<Element> // Shape: [output, input]
    var bias: Tensor<Element>? // Shape: [output]

    // Cache for forward input
    private var cachedInput: Tensor<Element>?

    init(inputDim: Int, outputDim: Int, includeBias: Bool = true) {
        weights = Tensor(zeros: [outputDim, inputDim]) // TODO: Add randomization
        bias = includeBias ? Tensor(zeros: [outputDim]) : nil // TODO: Add randomization
    }

    func forward(input: Tensor<Element>) -> Tensor<Element> {
        precondition(input.shape.last == weights.shape.last, "Input dimension must match weight's input_dim.")
        var output = input.matmul(weights.transposed())
        cachedInput = input
        if let b = bias?.broadcastTo(output.shape) {
            output.add(b)
        }
        return output
    }

    func backward(gradOutput: Tensor<Element>) -> Tensor<Element> {
        // Ensure the input is cached
        guard let input = self.cachedInput else {
            fatalError("No cached input. Did you forget to perform a forward pass?")
        }

        // Gradient with respect to input
        let gradInput = gradOutput.matmul(weights.transposed())

//        // Gradient with respect to weights
//        weights.grad.add(input.transposed().matmul(gradOutput))
//
//        // Gradient with respect to bias
//        bias.grad += gradOutput.sum(alongAxis: 0)

        // Clear cached input to free memory
        self.cachedInput = nil

        return gradInput
    }
}
