import Accelerate

extension Tensor {
    public func incremented(by scalar: Element) -> Self {
        let t = makeContiguous()
        let result = vDSP.add(scalar, t.dataSlice)
        return Self(shape, Array(result))
    }

    public static func +(lhs: Self, rhs: Element) -> Self {
        lhs.incremented(by: rhs)
    }

    public static func +(lhs: Element, rhs: Self) -> Self {
        rhs.incremented(by: lhs)
    }

    public static func +(lhs: Self, rhs: Self) -> Self {
        if lhs.shape == rhs.shape, lhs.isContiguous, rhs.isContiguous {
            let result = vDSP.multiply(addition: (lhs.dataSlice, rhs.dataSlice), 1)
            return Self(lhs.shape, result)
        }
        return performOperationSlow(lhs, rhs, +)
    }

    public mutating func add(_ other: Self) {
        precondition(shape == other.shape, "Shape mismatch")
        ensureUniquelyReferenced()

        if isContiguous {
            let other = other.makeContiguous()
            vDSP_vadd(
                dataSlice.baseAddress!,
                1,
                other.dataSlice.baseAddress!,
                1,
                dataSlice.baseAddress!,
                1,
                vDSP_Length(dataSlice.count)
            )
            return
        }

        forEachIndex { index in
            self[index] += other[index]
        }
    }

    public mutating func addBroadcasted(_ other: Self) {
        var other = other
        if shape != other.shape {
            guard let broadcastedOther = other.broadcastTo(shape) else {
                fatalError("Shapes cannot be broadcasted")
            }
            other = broadcastedOther
        }
        add(other)
    }
}
