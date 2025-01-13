extension Tensor {
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
}
