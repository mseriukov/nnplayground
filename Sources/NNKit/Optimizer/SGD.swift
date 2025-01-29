import Tensor

public class SGD: Optimizer {
    var parameters: [Parameter]
    let learningRate: Tensor.Element

    public init(parameters: [Parameter], learningRate: Tensor.Element) {
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
