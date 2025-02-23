import Tensor

public class MeanSquaredErrorOneHotActual: LossFunction {
    public init() {}

    public func forward(predicted: Tensor, actual: Tensor) -> Tensor {
        let actual = toOneHot(outputLen: predicted.shape[1], actual: actual)
        return ((predicted - actual) * (predicted - actual)).mean()
    }

    public func backward(predicted: Tensor, actual: Tensor) -> Tensor {
        let batchSize = actual.shape[0]
        let actual = toOneHot(outputLen: predicted.shape[1], actual: actual)
        return 2 * (predicted - actual) / Tensor(shape: [1], value: Tensor.Element(batchSize))
    }

    private func toOneHot(outputLen: Int, actual: Tensor) -> Tensor {
        var expected = Tensor.init(zeros: [actual.shape[0], outputLen])
        for i in 0..<actual.shape[0] {
            expected.assign(1.0, at: [i, Int(actual[i])])
        }
        return expected
    }
}
