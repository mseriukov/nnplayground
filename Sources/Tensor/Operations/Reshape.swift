extension Tensor {
    public enum ReshapeError: Error, CustomDebugStringConvertible {
        case sizeMismatch

        public var debugDescription: String {
            switch self {
            case .sizeMismatch: "Total size must remain the same when reshaping."
            }
        }

    }

    public func reshape(_ newShape: [Int]) throws -> Self {
        let newSize = newShape.reduce(1, *)
        let currentSize = shape.reduce(1, *)

        guard newSize == currentSize else {
            throw ReshapeError.sizeMismatch
        }        

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
}
