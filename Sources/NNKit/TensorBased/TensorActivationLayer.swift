import Tensor
import Accelerate

public class TensorActivationLayer<Element> where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    public enum Activation {
        case sigmoid
        case relu
        case softmax
    }

    public let activation: Activation

    private var cachedInput: Tensor<Element>?

    public init(_ activation: Activation) {
        self.activation = activation
    }

    public func forward(_ input: Tensor<Element>) -> Tensor<Element> {
        self.cachedInput = input

        let output: Tensor<Element>
        switch activation {
        case .sigmoid:
            output = input.map { Element(1.0 / (1.0 + exp(-Double($0)))) }
            return output
        default: break
            //        case .relu:
            //            output = Matrix(as: input, data: input.storage.map { max(0, $0) })
            //
            //        case .softmax:
            //            let expInput = exp(input - max(input))
            //            output = expInput / (expInput.storage.reduce(0.0, +))
            //        }
            //        self.output = output
            //        return output
        }
        return Tensor(zeros: [1])
    }

    public func backward(_ localGradient: Tensor<Element>) -> Tensor<Element> {
        guard var input = self.cachedInput else {
            fatalError("No cached input. Did you forget to perform a forward pass?")
        }
        switch activation {
        case .sigmoid:
            input = forward(input)
            input.mul((1 - input))
            return localGradient * input
        default: return Tensor(zeros: [1])
//        case .relu:
//            input = Matrix(as: input, data: input.storage.map { $0 > 0 ? 1.0 : 0.0 })
//            return elementwiseMul(
//                localGradient,
//                input
//            )
//        case .softmax:
//            let dSoftmax = Matrix.diagonal(from: output) - matmul(output.transposed(), output)
//            return matmul(localGradient, dSoftmax)
        }
    }
}
