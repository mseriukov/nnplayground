import Tensor

public protocol Optimizer {
    func step()
}

public class SGD: Optimizer {
    var parameters: [TensorParameter]
    let learningRate: Double

    public init(parameters: [TensorParameter], learningRate: Double) {
        self.parameters = parameters
        self.learningRate = learningRate
    }

    public func step() {
        for param in parameters {
            guard let grad = param.gradient else { continue }
            param.value = param.value - Tensor([], [learningRate]) * grad
        }
    }
}
