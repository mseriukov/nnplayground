import Tensor
import Accelerate

public class TensorActivationLayer: TensorLayer {
    public enum Activation {
        case sigmoid
        case relu
        case softmax
    }

    public let activation: Activation

    public let parameters: [TensorParameter] = []

    private var cachedInput: Tensor?
    private var cachedSoftmax: Tensor?

    public init(_ activation: Activation) {
        self.activation = activation
    }

    public func forward(_ input: Tensor) -> Tensor {
        self.cachedInput = input

        switch activation {
        case .sigmoid:
            return input.map { 1.0 / (1.0 + exp(-$0)) }

        case .relu:
            return input.map { max(0, $0) }

        case .softmax:
            let expInput = input.map { exp($0) }
            let sumExp = expInput.sum(alongAxis: expInput.shape.count - 1, keepDims: true)
            let softmaxOutput = expInput / sumExp
            self.cachedSoftmax = softmaxOutput
            return softmaxOutput
        }
    }

    public func backward(_ localGradient: Tensor) -> Tensor {
        guard var input = self.cachedInput else {
            fatalError("No cached input. Did you forget to perform a forward pass?")
        }
        switch activation {
        case .sigmoid:
            input = forward(input)
            input.mul((1 - input))
            return localGradient * input

        case .relu:
            input = input.map { $0 > 0 ? 1.0 : 0.0 }
            return localGradient * input

        case .softmax:
            guard let cachedSoftmax else {
                fatalError("No cached output. Did you forget to perform a forward pass?")
            }
            // Gradient calculation using cached softmax
            let sGrad = localGradient * cachedSoftmax
            let sumGrad = sGrad.sum(alongAxis: sGrad.shape.count - 1, keepDims: true)
            return sGrad - cachedSoftmax * sumGrad
        }
    }
}
