public struct ReshapedView<Element, Parent: NDArrayType>: NDArrayType where Parent.Element == Element  {
    public var parent: Parent
    public var storage: Parent.Storage { parent.storage }
    public var shape: Shape

    public init(parent: Parent, shape: Shape) {
        precondition(parent.shape.size == shape.size, "Shapes doesn't match.")
        self.parent = parent
        self.shape = shape
    }

    public subscript(s: [Int]) -> Element {
        get {
            storage[shape.flatIndex(with: s)]
        }
        set {
            storage[shape.flatIndex(with: s)] = newValue
        }
    }
}
