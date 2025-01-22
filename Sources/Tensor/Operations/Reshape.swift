extension Tensor {
    public func reshape(_ newShape: [Int]) -> Self {
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
}
