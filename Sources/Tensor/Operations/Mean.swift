extension Tensor where Element: BinaryFloatingPoint {
    public func mean() -> Self {
        Self(shape: [], value: sum().value / Element(shape.reduce(1, *)))
    }
}
