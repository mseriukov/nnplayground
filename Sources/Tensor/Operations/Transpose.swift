extension Tensor {
    public func transposed() -> Self {
        precondition(shape.count == 2, "Must be 2D for now.")
        return Self(
            storage: storage,
            shape: shape.reversed(),
            strides: strides.reversed()
        )
    }
}
