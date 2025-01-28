extension Tensor {
    public mutating func mapInPlace(_ transform: (inout Element) -> Void) {
        storage.buffer.mapInPlace(transform)
    }

    public func map(_ transform: (Element) -> Element) -> Self {
        var copy = copy()
        copy.mapInPlace { $0 = transform($0) }
        return copy
    }
}

extension MutableCollection {
    mutating func mapInPlace(_ x: (inout Element) -> ()) {
        for i in indices {
            x(&self[i])
        }
    }
}
