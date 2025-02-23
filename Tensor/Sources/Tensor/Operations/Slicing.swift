extension Tensor {
    public func slice(start: [Int], shape: [Int]) -> Self {
        var newTensor = self
        newTensor.shape = shape
        newTensor.offset = flatIndex(start)
        return newTensor
    }
}
