extension Tensor {
    public func mean() -> Self {
        Self(shape: [], value: sum().value / Double(shape.reduce(1, *)))
    }
}
