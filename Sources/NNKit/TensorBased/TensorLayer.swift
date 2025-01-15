import Tensor

public protocol TensorLayer<Element> {
    associatedtype Element where Element: BinaryFloatingPoint, Element.RawSignificand: FixedWidthInteger

    var parameters: [TensorParameter<Element>] { get }

    func forward(_ input: Tensor<Element>) -> Tensor<Element>
    func backward(_ localGradient: Tensor<Element>) -> Tensor<Element>
}
