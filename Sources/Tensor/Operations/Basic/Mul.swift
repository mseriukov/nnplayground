import Accelerate

extension Tensor {
    public mutating func mul(_ other: Self) {
        precondition(shape == other.shape, "Shape mismatch")
        ensureUniquelyReferenced()

        if isContiguous {
            let other = other.makeContiguous()
            vDSP_vmul(
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
            self[index] *= other[index]
        }
    }

    public func multiplied(by scalar: Element) -> Self {
        let t = makeContiguous()
        let result = vDSP.multiply(scalar, t.dataSlice)
        return Self(shape, Array(result))
    }

    public mutating func mulBroadcasted(_ other: Self) {
        var other = other
        if shape != other.shape {
            guard let broadcastedOther = other.broadcastTo(shape) else {
                fatalError("Shapes cannot be broadcasted")
            }
            other = broadcastedOther
        }
        mul(other)
    }

    public static func *(lhs: Element, rhs: Self) -> Self {
        rhs.multiplied(by: lhs)
    }

    public static func *(lhs: Self, rhs: Element) -> Self {
        lhs.multiplied(by: rhs)
    }

    public static func *(lhs: Self, rhs: Self) -> Self {
        if lhs.shape == rhs.shape, lhs.isContiguous, rhs.isContiguous {
            let result = vDSP.multiply(lhs.dataSlice, rhs.dataSlice)
            return Self(lhs.shape, result)
        }
        return performOperationSlow(lhs, rhs, *)
    }
}

