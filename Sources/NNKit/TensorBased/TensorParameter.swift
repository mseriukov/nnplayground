import Tensor

public class TensorParameter<Element: BinaryFloatingPoint> {
    public var weights: Tensor<Element>
    public var gradient: Tensor<Element>?
    public internal(set) var name: String?

    public init(tensor: Tensor<Element>, name: String? = nil) {
        self.weights = tensor
        self.gradient = nil
        self.name = name
    }

    public func resetGrad() {
        gradient = nil
    }
}
