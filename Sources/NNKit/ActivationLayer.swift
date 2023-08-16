import AlgebraKit
import Accelerate

public class ActivationLayer: Layer {
    public let activation: Activation

    private(set) public var input: Matrix = .zero
    private(set) public var output: Matrix = .zero

    public init(
        _ activation: Activation
    ) {
        self.activation = activation
    }

    public func forward(_ input: Matrix) -> Matrix {
        self.input = input
        return activation.forward(input)
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        Matrix.elementwiseMul(m1: localGradient, m2: activation.backward(input))
    }

    // FYI: eta is just typable version of η.
    public func updateParameters(eta: Float) {
        // Static layer
    }
}
