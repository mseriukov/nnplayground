import AlgebraKit

public class Conv2DLayer: Layer {
    public var input: Matrix = .zero
    public var output: Matrix = .zero
    public var parameters: [Parameter] = []

    public func forward(_ input: Matrix) -> Matrix {
        .zero
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        .zero
    }
}
