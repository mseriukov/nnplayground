import Tensor

public class TensorParameter<Element> where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    public var value: Tensor<Element>
    public var gradient: Tensor<Element>?
    public internal(set) var name: String?

    public init(tensor: Tensor<Element>, name: String? = nil) {
        self.value = tensor
        self.gradient = nil
        self.name = name
    }

    public func resetGrad() {
        gradient = nil
    }

    public func randomize(_ generator: inout RandomNumberGenerator) {
        value = Tensor.random(
            shape: value.shape,
            distribution: .kaiming(channels: value.shape.last!),
            generator: &generator
        )
    }
}
