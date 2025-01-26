public struct Tensor {
    public typealias Element = Float32
    public internal(set) var size: Int
    public internal(set) var shape: [Int]
    public internal(set) var strides: [Int]
    public internal(set) var offset: Int
    public internal(set) var storage: TensorStorage

    public var value: Element {
        storage[0]
    }

    public var rank: Int {
        shape.count
    }

    public init(
        storage: TensorStorage,
        shape: [Int],
        strides: [Int]? = nil,
        offset: Int = 0
    ) {
        self.storage = storage
        self.shape = shape
        self.size = shape.reduce(1, *)
        self.offset = offset

        self.strides = if let strides {
            strides
        } else {
            shape
                .reversed()
                .dropLast()
                .reduce(Array<Int>([1])) { [$0.first! * $1] + $0 }
        }
    }

    public init(_ shape: [Int], _ data: [Element]) {
        precondition(shape.reduce(1, *) == data.count, "Data doesn't follow shape.")
        self.init(storage: TensorStorage(data), shape: shape)
    }

    public init(shape: [Int], value: Element) {
        let size = shape.reduce(1, *)
        let storage = TensorStorage(Array<Element>(repeating: value, count: size))
        self.init(storage: storage, shape: shape)
    }

    public init(zeros shape: [Int]) {
        self.init(shape: shape, value: 0.0)
    }

    public var isContiguous: Bool {
        if size < storage.size {
            return false
        }
        var expectedStride = 1
        for i in (0..<shape.count).reversed() {
            if strides[i] != expectedStride {
                return false
            }
            expectedStride *= shape[i]
        }
        return true
    }

    public var isScalar: Bool {
        return shape == [] || shape == [1]
    }

    public subscript(_ index: [Int]) -> Element {
        get {
            return storage[flatIndex(index)]
        }
        set {
            ensureUniquelyReferenced()
            storage[flatIndex(index)] = newValue
        }
    }

    public subscript(_ s: Int...) -> Element {
        self[s]
    }

    public mutating func assign(_ value: Element, at index: [Int]) {
        storage[flatIndex(index)] = value
    }

    public func makeContiguous() -> Self {
        if isContiguous {
            return self
        }

        let newStorage = TensorStorage(size: shape.reduce(1, *))
        var newDataIndex = 0

        let iterator = TensorIndexSequence(shape: shape)

        for indices in iterator {
            let flatIndex = offset + zip(indices, strides).map(*).reduce(0, +)
            newStorage.data[newDataIndex] = storage.data[flatIndex]
            newDataIndex += 1
        }

        return Self(storage: newStorage, shape: shape)
    }

    mutating func ensureUniquelyReferenced() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = storage.copy()
        }
    }

    func flatIndex(_ indicies: [Int]) -> Int {
        precondition(indicies.count == shape.count, "Indicies and shape mismatch")
        return zip(indicies, strides).reduce(0, { $0 + $1.0 * $1.1 }) + offset
    }

    public func forEachIndex(_ closure: ([Int]) -> Void) {
        for index in TensorIndexSequence(shape: shape) {
            closure(index)
        }
    }

    static func defaultStrides(for shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        for i in (0..<(shape.count - 1)).reversed() {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        return strides
    }
}

extension Tensor: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Element) {
        self.init(shape: [1], value: value)
    }
}

extension Tensor: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init(shape: [1], value: Element(value))
    }
}
