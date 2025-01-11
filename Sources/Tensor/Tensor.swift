import Accelerate

public struct Tensor<Element: BinaryFloatingPoint> {
    public typealias Storage = TensorStorage<Element>
    public let size: Int
    public private(set) var shape: [Int]
    public private(set) var strides: [Int]
    public private(set) var offset: Int
    public private(set) var storage: Storage
    public private(set) var gradientStorage: Storage?

    public init(
        storage: Storage,
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

    mutating func zeroGradient() {
        // Initialize gradient storage with zeros if not already present
        gradientStorage = TensorStorage(size: storage.data.count, initialValue: 0.0)
    }

    func gradient() -> Tensor? {
        guard let gradientStorage else { return nil }
        return Tensor(storage: gradientStorage, shape: shape)
    }

    var isContiguous: Bool {
        var expectedStride = 1
        for i in (0..<shape.count).reversed() {
            if strides[i] != expectedStride {
                return false
            }
            expectedStride *= shape[i]
        }
        return true
    }

    var isScalar: Bool {
        return shape == [] || shape == [1]
    }

    public func makeContiguous() -> Self {
        if isContiguous {
            return self  // Already contiguous
        }

        let newStorage = Storage(size: shape.reduce(1, *))
        var newDataIndex = 0

        let iterator = TensorIndexSequence(shape: shape)

        for indices in iterator {
            let flatIndex = offset + zip(indices, strides).map(*).reduce(0, +)
            newStorage.data[newDataIndex] = storage.data[flatIndex]
            newDataIndex += 1
        }

        return Self(storage: newStorage, shape: shape)
    }

    private mutating func ensureUniquelyReferenced() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = storage.copy()
        }
    }

    public subscript(_ s: [Int]) -> Element {
        get {
            let index = flatIndex(with: s) + offset
            return storage[index]
        }
        set {
            ensureUniquelyReferenced()
            let index = flatIndex(with: s) + offset
            storage[index] = newValue
        }
    }

    public subscript(_ s: Int...) -> Element {
        self[s]
    }

    private func flatIndex(with indicies: [Int]) -> Int {
        precondition(indicies.count == shape.count, "Indicies and shape mismatch")
        return zip(indicies, strides).reduce(0, { $0 + $1.0 * $1.1 }) + offset
    }

    func reshape(_ newShape: [Int]) -> Self {
        let newSize = newShape.reduce(1, *)
        let currentSize = shape.reduce(1, *)

        precondition(newSize == currentSize, "Total size must remain the same when reshaping.")

        if isContiguous {
            return Self(
                storage: storage,
                shape: newShape,
                strides: Self.defaultStrides(for: newShape),
                offset: offset
            )
        }

        // Copy if non-contiguous
        let contiguousTensor = makeContiguous()
        return Self(
            storage: contiguousTensor.storage,
            shape: newShape,
            strides: Self.defaultStrides(for: newShape),
            offset: 0
        )
    }

    public func slice(start: [Int], size: [Int]) -> Self {
        var newTensor = self
        newTensor.shape = size
        newTensor.offset = flatIndex(with: start)
        return newTensor
    }
}

extension Tensor {
    private static func defaultStrides(for shape: [Int]) -> [Int] {
        DefaultStridesGenerator.defaultStrides(for: shape)
    }

    private func broadcastShapes(_ shape1: [Int], _ shape2: [Int]) -> [Int]? {
        let maxRank = max(shape1.count, shape2.count)
        var resultShape = [Int](repeating: 1, count: maxRank)

        for i in 0..<maxRank {
            let dim1 = i < maxRank - shape1.count ? 1 : shape1[i - (maxRank - shape1.count)]
            let dim2 = i < maxRank - shape2.count ? 1 : shape2[i - (maxRank - shape2.count)]

            if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
                resultShape[i] = max(dim1, dim2)
            } else {
                return nil  // Shapes are incompatible
            }
        }
        return resultShape
    }

    func broadcastTo(_ shape: [Int]) -> Self? {
        if isScalar {
            var scalarView = self
            scalarView.shape = shape
            scalarView.strides = Array(repeating: 0, count: shape.count)
            return scalarView
        }

        guard let newShape = broadcastShapes(self.shape, shape) else {
            return nil
        }

        var newStrides = [Int](repeating: 0, count: newShape.count)
        var offsetShape = self.shape

        // Expand dimensions to match new shape
        while offsetShape.count < newShape.count {
            offsetShape.insert(1, at: 0)
        }

        for i in 0..<newShape.count {
            if offsetShape[i] == newShape[i] {
                newStrides[i] = strides[i - (newShape.count - strides.count)]
            } else if offsetShape[i] == 1 {
                newStrides[i] = 0  // Broadcast: same value across dimension
            } else {
                return nil
            }
        }

        var broadcastedTensor = self
        broadcastedTensor.shape = newShape
        broadcastedTensor.strides = newStrides
        return broadcastedTensor
    }
}

extension Tensor {
    public mutating func add(_ other: Self) {
        precondition(shape == other.shape, "Shape mismatch")
        ensureUniquelyReferenced()

        for index in TensorIndexSequence(shape: shape) {
            self[index] += other[index]
        }
    }

    public mutating func addBroadcasted(_ other: Self) {
        var other = other
        if shape != other.shape {
            guard let broadcastedOther = other.broadcastTo(shape) else {
                fatalError("Shapes cannot be broadcasted")
            }
            other = broadcastedOther
        }
        add(other)
    }
}

extension Tensor {
    public mutating func mul(_ other: Self) {
        precondition(shape == other.shape, "Shape mismatch")
        ensureUniquelyReferenced()

        for index in TensorIndexSequence(shape: shape) {
            self[index] *= other[index]
        }
    }

    public mutating func mulBroadcasted(_ other: Self) {
        var other = other
        if shape != other.shape {
            guard let broadcastedOther = other.broadcastTo(shape) else {
                fatalError("Shapes cannot be broadcasted")
            }
            other = broadcastedOther
        }
        mul(other)
    }
}
