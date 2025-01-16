import Tensor

public protocol Optimizer<Element> {
    associatedtype Element where Element: BinaryFloatingPoint, Element.RawSignificand: FixedWidthInteger

    func step()
}

public class SGD<Element>: Optimizer where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    var parameters: [TensorParameter<Element>]
    let learningRate: Element

    public init(parameters: [TensorParameter<Element>], learningRate: Element) {
        self.parameters = parameters
        self.learningRate = learningRate
    }

    public func step() {
        for param in parameters {
            guard let grad = param.gradient else { continue }
            param.value = param.value - Tensor<Element>([], [learningRate]) * grad
        }
    }
}
