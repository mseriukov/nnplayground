public final class TensorStorage {
    public var data: [Double]

    public var size: Int {
        data.count
    }

    public init(_ data: [Double]) {
        self.data = data
    }

    public convenience init(
        size: Int,
        initialValue: Double = 0.0
    ) {
        self.init(Array(repeating: initialValue, count: size))
    }

    public func copy() -> TensorStorage {
        let newStorage = TensorStorage(size: data.count)
        newStorage.data = data
        return newStorage
    }

    public subscript(_ index: Int) -> Double {
        get {
            data[index]
        }
        set {
            data[index] = newValue
        }
    }
}

extension TensorStorage: ExpressibleByArrayLiteral {
    public convenience init(arrayLiteral elements: Double...) {
        self.init(elements)
    }
}
