import Tensor

protocol LossFunction<Element> {
    associatedtype Element where Element: BinaryFloatingPoint, Element.RawSignificand: FixedWidthInteger

    func forward(predicted: Tensor<Element>, actual: Tensor<Element>) -> Tensor<Element>
    func backward(predicted: Tensor<Element>, actual: Tensor<Element>) -> Tensor<Element>
}
