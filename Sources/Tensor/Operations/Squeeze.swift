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

    public mutating func unsqueeze(axis: Int) {
        assert(axis >= 0 && axis <= self.shape.count, "Axis out of bounds.")
        var newShape = self.shape
        var newStrides = self.strides

        newShape.insert(1, at: axis)

        // If contiguous, stride should be 0 for broadcasting.
        // If non-contiguous (sliced), inherit closest valid stride.
        let inheritedStride = (axis > 0) ? newStrides[axis - 1] : newStrides.first ?? 1
        newStrides.insert(isContiguous ? 0 : inheritedStride, at: axis)

        shape = newShape
        strides = newStrides
    }

    public func unsqueezed(axis: Int) -> Self {
        var result = self
        result.unsqueeze(axis: axis)
        return result
    }
}
