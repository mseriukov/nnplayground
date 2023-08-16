import AlgebraKit

public protocol Layer {
    var input: Matrix { get }
    var output: Matrix { get }

    func forward(_ input: Matrix) -> Matrix
    func backward(_ localGradient: Matrix) -> Matrix
    func updateParameters(eta: Float)
}
