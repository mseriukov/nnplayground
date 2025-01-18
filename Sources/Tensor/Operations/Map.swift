extension Tensor {
    public mutating func mapInPlace(_ transform: (inout Double) -> Void) {
        storage.data.mapInPlace(transform)
    }

    public func map(_ transform: (Double) -> Double) -> Self {
        var tensor = self
        tensor.ensureUniquelyReferenced()
        tensor.storage.data.mapInPlace { $0 = transform($0) }
        return tensor
    }
}

extension MutableCollection {
    mutating func mapInPlace(_ x: (inout Element) -> ()) {
        for i in indices {
            x(&self[i])
        }
    }
}
