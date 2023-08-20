import AlgebraKit
import Accelerate

public enum Activation {
    case sigmoid
    case relu
    case softmax
}

// This is probably slow as hell.
public class ActivationLayer: Layer {
    public let activation: Activation
    private(set) public var input: Matrix = .zero
    private(set) public var output: Matrix = .zero

    public var parameters: [Parameter] {
        []
    }

    public init(_ activation: Activation) {
        self.activation = activation
    }

    public func forward(_ input: Matrix) -> Matrix {
        self.input = input

        let output: Matrix
        switch activation {
        case .sigmoid:
            output = Matrix(as: input, data: input.storage.map { 1.0 / (1.0 + exp(-$0)) })

        case .relu:
            output = Matrix(as: input, data: input.storage.map { max(0, $0) })

        case .softmax:
            let expInput = exp(input - max(input))
            output = expInput / (expInput.storage.reduce(0.0, +))
        }
        self.output = output
        return output
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        var input = input
        switch activation {
        case .sigmoid:
            input = elementwiseMul(forward(input), (Matrix(as: input, repeating: 1) - forward(input)))
            return elementwiseMul(localGradient, input)
        case .relu:
            input = Matrix(as: input, data: input.storage.map { $0 > 0 ? 1.0 : 0.0 })
            return elementwiseMul(
                localGradient,
                input
            )

        case .softmax:
            let dSoftmax = Matrix.diagonal(from: output) - matmul(output.transposed(), output)
            return matmul(localGradient, dSoftmax)
        }

    }

    // FYI: eta is just a typeable version of η.
    public func updateParameters(eta: Float) {
        // Static layer
    }
}
