import Tensor

public protocol Layer {
    var parameters: [Parameter] { get }

    func forward(_ input: Tensor) throws -> Tensor
    func backward(_ localGradient: Tensor) throws -> Tensor
}
