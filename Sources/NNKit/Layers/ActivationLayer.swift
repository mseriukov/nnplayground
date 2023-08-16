import AlgebraKit
import Accelerate

public enum Activation {
    case sigmoid
    case relu
}

// This is probably slow as hell.
public class ActivationLayer: Layer {
    public let activation: Activation
    private(set) public var input: Matrix = .zero
    private(set) public var output: Matrix = .zero

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
        }
        self.output = output
        return output
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        var input = input
        switch activation {
        case .sigmoid:
            input = Matrix.elementwiseMul(
                m1: forward(input),
                m2:  (Matrix(as: input, repeating: 1) - forward(input))
            )
            return Matrix.elementwiseMul(m1: localGradient, m2: input)
        case .relu:
            input = Matrix(as: input, data: input.storage.map { $0 > 0 ? 1.0 : 0.0 })
            return Matrix.elementwiseMul(m1: localGradient, m2: input)
        }

    }

    // FYI: eta is just a typeable version of η.
    public func updateParameters(eta: Float) {
        // Static layer
    }
}
