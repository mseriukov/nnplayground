import Tensor

public protocol LossFunction {
    func forward(predicted: Tensor, actual: Tensor) -> Tensor
    func backward(predicted: Tensor, actual: Tensor) -> Tensor
}
