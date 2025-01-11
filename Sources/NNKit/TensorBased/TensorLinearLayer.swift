import Tensor

public class TensorLinearLayer<Element: BinaryFloatingPoint> {
    var weight: Tensor<Element> // Shape: [output, input]
    var bias: Tensor<Element>? // Shape: [output]

    init(inputDim: Int, outputDim: Int, includeBias: Bool = true) {
        weight = Tensor(zeros: [outputDim, inputDim]) // TODO: Add randomization
        bias = includeBias ? Tensor(zeros: [outputDim]) : nil // TODO: Add randomization
    }

    func forward(input: Tensor<Element>) -> Tensor<Element> {
        precondition(input.shape.last == weight.shape.last, "Input dimension must match weight's input_dim.")
        var output = input.matmul(weight.transposed())
        if let b = bias?.broadcastTo(output.shape) {
            output.add(b)
        }
        return output
    }
}
