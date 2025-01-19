public final class TensorStorage {
    public var data: [Tensor.Element]

    public var size: Int {
        data.count
    }

    public init(_ data: [Tensor.Element]) {
        self.data = data
    }

    public convenience init(
        size: Int,
        initialValue: Tensor.Element = 0.0
    ) {
        self.init(Array(repeating: initialValue, count: size))
    }

    public func copy() -> TensorStorage {
        let newStorage = TensorStorage(size: data.count)
        newStorage.data = data
        return newStorage
    }

    public subscript(_ index: Int) -> Tensor.Element {
        get {
            data[index]
        }
        set {
            data[index] = newValue
        }
    }
}

extension TensorStorage: ExpressibleByArrayLiteral {
    public convenience init(arrayLiteral elements: Tensor.Element...) {
        self.init(elements)
    }
}
