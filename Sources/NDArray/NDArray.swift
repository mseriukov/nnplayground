public struct NDArray<Element, Storage>: NDArrayType where Storage: LinearStorageType, Storage.Element == Element {
    public private(set) var storage: Storage
    public private(set) var shape: Shape

    public init(storage: Storage, shape: Shape) {
        self.storage = storage
        self.shape = shape
    }

    public subscript(_ s: [Int]) -> Element {
        get {
            storage[shape.flatIndex(with: s)]
        }
        set {
            storage[shape.flatIndex(with: s)] = newValue
        }
    }
}
