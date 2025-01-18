import Tensor

public class TensorLinearLayer: TensorLayer  {
    var weights: TensorParameter // Shape: [output, input]
    var bias: TensorParameter? // Shape: [output]

    public var parameters: [TensorParameter] {
        [weights, bias].compactMap { $0 }
    }

    private var cachedInput: Tensor?
    private var randomGenerator: any RandomNumberGenerator = SystemRandomNumberGenerator()

    public init(inputDim: Int, outputDim: Int, includeBias: Bool = true) {
        weights = TensorParameter(tensor: Tensor.random(
            shape: [inputDim,
                    outputDim],
            distribution: .kaiming(channels: inputDim),
            generator: &randomGenerator
        ))
        bias = includeBias ? TensorParameter(tensor: Tensor.random(
            shape: [outputDim],
            distribution: .kaiming(channels: inputDim),
            generator: &randomGenerator
        )) : nil
    }

    public func forward(_ input: Tensor) -> Tensor {
        var input = input
        // Ensure we have batch dimension.
        if input.rank == 1 {
            input.unsqueeze(axis: 0)
        }
        precondition(input.shape.last == weights.value.shape.first, "Input dimension must match weight's input_dim.")
        var output = input.matmul(weights.value)
        cachedInput = input
        if let b = bias?.value.broadcastTo(output.shape) {
            output.add(b)
        }
        return output
    }

    public func backward(_ localGradient: Tensor) -> Tensor {
        guard let input = self.cachedInput else {
            fatalError("No cached input. Did you forget to perform a forward pass?")
        }
        if weights.gradient == nil {
            weights.gradient = Tensor(zeros: weights.value.shape)
        }

        if let bias, bias.gradient == nil {
            bias.gradient = Tensor(zeros: bias.value.shape)
        }

        let gradInput = localGradient.matmul(weights.value.transposed())

        weights.gradient?.add(input.transposed().matmul(localGradient))
        bias?.gradient?.add(localGradient.sum(alongAxis: 0))

        cachedInput = nil
        return gradInput
    }
}
