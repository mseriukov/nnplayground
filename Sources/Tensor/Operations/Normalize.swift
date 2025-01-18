extension Tensor {
    public mutating func normalize() {
        storage.data.normalize()
    }

    public func normalized() -> Self {
        var tensor = self
        tensor.normalize()        
        return tensor
    }
}
