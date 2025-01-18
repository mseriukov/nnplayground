import Tensor

public class MeanSquaredError: LossFunction {
    public init() {}

    public func forward(predicted: Tensor, actual: Tensor) -> Tensor {
        ((predicted - actual) * (predicted - actual)).mean()
    }

    public func backward(predicted: Tensor, actual: Tensor) -> Tensor {
        let batchSize = actual.shape[0]
        return 2 * (predicted - actual) / Tensor(shape: [1], value: Double(batchSize))
    }
}
