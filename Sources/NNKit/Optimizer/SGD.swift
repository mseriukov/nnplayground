import Tensor

public class SGD: Optimizer {
    var parameters: [TensorParameter]
    let learningRate: Tensor.Element

    public init(parameters: [TensorParameter], learningRate: Tensor.Element) {
        self.parameters = parameters
        self.learningRate = learningRate
    }

    public func step() {
        for param in parameters {
            guard let grad = param.gradient else { continue }
            param.value = param.value - grad * learningRate
        }
    }
}
