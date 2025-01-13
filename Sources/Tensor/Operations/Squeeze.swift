extension Tensor {
    public mutating func squeeze(_ axes: [Int] = []) {
        var newShape: [Int] = []
        var newStrides: [Int] = []
        var axes = axes
        if axes.isEmpty {
            axes = Array(0..<shape.count)
        }
        for (axis, size) in shape.enumerated() {
            if size != 1 || !axes.contains(axis) {
                newShape.append(size)
                newStrides.append(strides[axis])
            }
        }
        shape = newShape
        strides = newStrides
    }

    public func squeezed(_ axes: [Int] = []) -> Self {
        var result = self
        result.squeeze(axes)
        return result
    }

    public func unsqueeze(axis: Int) -> Self {
        assert(axis >= 0 && axis <= self.shape.count, "Axis out of bounds.")

        var newShape = self.shape
        var newStrides = self.strides

        newShape.insert(1, at: axis)
        newStrides.insert(0, at: axis) // Stride of 0 since it's a singleton dimension

        return Self(storage: self.storage, shape: newShape, strides: newStrides, offset: self.offset)
    }
}
