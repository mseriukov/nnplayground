public final class TensorStorage<Element: BinaryFloatingPoint> {
    var data: [Element]

    public init(_ data: [Element]) {
        self.data = data
    }

    public convenience init(
        size: Int,
        initialValue: Element = 0.0
    ) {
        self.init(Array(repeating: initialValue, count: size))
    }

    public func copy() -> TensorStorage {
        let newStorage = TensorStorage(size: data.count)
        newStorage.data = data
        return newStorage
    }

    public subscript(_ index: Int) -> Element {
        get {
            data[index]
        }
        set {
            data[index] = newValue
        }
    }
}
