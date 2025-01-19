import Tensor

public protocol TensorLayer {
    var parameters: [TensorParameter] { get }

    func forward(_ input: Tensor) -> Tensor
    func backward(_ localGradient: Tensor) -> Tensor
}
