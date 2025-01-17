import Tensor

public class MeanSquaredError<Element>: LossFunction where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    public init() {}

    public func forward(predicted: Tensor<Element>, actual: Tensor<Element>) -> Tensor<Element> {
        ((predicted - actual) * (predicted - actual)).mean()
    }

    public func backward(predicted: Tensor<Element>, actual: Tensor<Element>) -> Tensor<Element> {
        2 * (predicted - actual) / Tensor(shape: [1], value: Element(actual.shape.reduce(1, *)))
    }
}
