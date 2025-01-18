import Tensor

public class TensorParameter {
    public var value: Tensor
    public var gradient: Tensor?
    public internal(set) var name: String?

    public init(tensor: Tensor, name: String? = nil) {
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
