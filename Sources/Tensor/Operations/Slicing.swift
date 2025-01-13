extension Tensor {
    public func slice(start: [Int], shape: [Int]) -> Self {
        var newTensor = self
        newTensor.shape = shape
        newTensor.offset = flatIndex(start)
        newTensor.size = shape.reduce(1, *)
        return newTensor
    }
}
