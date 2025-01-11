import Accelerate

public struct Tensor<Element: BinaryFloatingPoint> {
    public typealias Storage = TensorStorage<Element>
    public private(set) var size: Int
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

    init(shape: [Int], value: Element) {
        let size = shape.reduce(1, *)
        let storage = TensorStorage(Array<Element>(repeating: value, count: size))
        self.init(storage: storage, shape: shape)
    }

    init(zeros shape: [Int]) {
        self.init(shape: shape, value: 0.0)
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
        if size < storage.size {
            return false
        }
        if offset != 0 {
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

    public subscript(_ index: [Int]) -> Element {
        get {
            return storage[flatIndex(index)]
        }
        set {
            ensureUniquelyReferenced()
            storage[flatIndex(index)] = newValue
        }
    }

    public mutating func assign(_ value: Element, at index: [Int]) {
        storage[flatIndex(index)] = value
    }

    public subscript(_ s: Int...) -> Element {
        self[s]
    }

    private func flatIndex(_ indicies: [Int]) -> Int {
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

    public func slice(start: [Int], shape: [Int]) -> Self {
        var newTensor = self
        newTensor.shape = shape
        newTensor.offset = flatIndex(start)
        newTensor.size = shape.reduce(1, *)
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

    public func broadcastTo(_ shape: [Int]) -> Self? {
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

    public func forEachIndex(_ closure: ([Int]) -> Void) {
        for index in TensorIndexSequence(shape: shape) {
            closure(index)
        }
    }

    public mutating func assign(start: [Int], tensor: Self) {
        // TODO: verify preconditions.
        precondition(start.count == shape.count, "Slice must have the same number of dimensions as the tensor.")
        for (start, end) in zip(start, tensor.shape) {
            precondition(start + end <= self.shape[0], "Slice exceeds tensor dimensions.")
        }

        var slicedView = self.slice(
            start: start,
            shape: tensor.shape
        )
        slicedView.forEachIndex { index in
            slicedView.assign(tensor[index], at: index)
        }
    }

    public func matmul(_ other: Self) -> Self {
        precondition(shape.count == 2 && other.shape.count == 2, "Both tensors must be 2D for matmul.")
        precondition(shape[1] == other.shape[0], "Inner dimensions must match for matmul.")

        let m = shape[0]
        let n = shape[1]
        let p = other.shape[1]

        let result = Self(zeros: [m, p])
        result.forEachIndex { resultIndex in
            let i = resultIndex[0]
            let j = resultIndex[1]

            var sum: Element = 0.0
            for k in 0..<n {
                let a = self.flatIndex([i, k])
                let b = other.flatIndex([k, j])
                sum += self.storage[a] * other.storage[b]
            }
            result.storage[result.flatIndex(resultIndex)] = sum
        }
        return result
    }
}



extension Tensor {
    func unsqueeze(axis: Int) -> Self {
        assert(axis >= 0 && axis <= self.shape.count, "Axis out of bounds.")

        var newShape = self.shape
        var newStrides = self.strides

        newShape.insert(1, at: axis)
        newStrides.insert(0, at: axis) // Stride of 0 since it's a singleton dimension

        return Self(storage: self.storage, shape: newShape, strides: newStrides, offset: self.offset)
    }
}

extension Tensor {
    func batchedMatMul(_ A: Self, _ B: Self) -> Self {
        assert(A.shape.count == 3 && B.shape.count == 3, "Both tensors must be 3D.")
        assert(A.shape[0] == B.shape[0], "Batch sizes must match.")
        assert(A.shape[2] == B.shape[1], "Inner dimensions must match.")

        let batchSize = A.shape[0]
        let m = A.shape[1]
        let n = A.shape[2]
        let p = B.shape[2]

        var result = Tensor(zeros: [batchSize, m, p])

        for i in 0..<batchSize {
            let a = A.slice(start: [i, 0, 0], shape: [1, m, n]).squeeze(axis: 0)
            let b = B.slice(start: [i, 0, 0], shape: [1, n, p]).squeeze(axis: 0)
            let c = a.matmul(b)
            result.assign(start: [i, 0, 0], tensor: c.unsqueeze(axis: 0))
        }

        return result
    }
}

extension Tensor {
    func squeeze(axis: Int? = nil) -> Self {
        var newShape = self.shape
        var newStrides = self.strides

        if let axis = axis {
            assert(axis >= 0 && axis < shape.count, "Axis out of bounds.")
            assert(shape[axis] == 1, "Cannot squeeze a dimension with size greater than 1.")

            newShape.remove(at: axis)
            newStrides.remove(at: axis)
        } else {
            // Squeeze all dimensions with size 1
            for (dim, size) in shape.enumerated().reversed() {
                if size == 1 {
                    newShape.remove(at: dim)
                    newStrides.remove(at: dim)
                }
            }
        }

        return Self(storage: self.storage, shape: newShape, strides: newStrides, offset: self.offset)
    }
}

extension Tensor {
    public mutating func add(_ other: Self) {
        precondition(shape == other.shape, "Shape mismatch")
        ensureUniquelyReferenced()

        forEachIndex { index in
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

        forEachIndex { index in
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
