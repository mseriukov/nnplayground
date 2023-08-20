import AlgebraKit

public protocol Layer {
    var input: Matrix { get }
    var output: Matrix { get }
    var parameters: [Parameter] { get }

    func forward(_ input: Matrix) -> Matrix
    func backward(_ localGradient: Matrix) -> Matrix
}
