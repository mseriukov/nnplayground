import Tensor

public protocol Layer {
    var parameters: [Parameter] { get }

    func forward(_ input: Tensor) -> Tensor
    func backward(_ localGradient: Tensor) -> Tensor
}
