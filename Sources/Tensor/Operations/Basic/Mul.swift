import Accelerate

extension Tensor {
    public mutating func mul(_ other: Self) {
        precondition(shape == other.shape, "Shape mismatch")
        ensureUniquelyReferenced()

        if isContiguous {
            let other = other.makeContiguous()
            vDSP_vmulD(
                storage.data,
                1,
                other.storage.data,
                1,
                &storage.data,
                1,
                vDSP_Length(storage.data.count)
            )
            return
        }

        forEachIndex { index in
            self[index] *= other[index]
        }
    }

    public func multiplied(by scalar: Double) -> Self {
        let t = makeContiguous()
        let result = vDSP.multiply(scalar, t.storage.data)
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

    public static func *(lhs: Double, rhs: Self) -> Self {
        rhs.multiplied(by: lhs)
    }

    public static func *(lhs: Self, rhs: Double) -> Self {
        lhs.multiplied(by: rhs)
    }

    public static func *(lhs: Self, rhs: Self) -> Self {
        if lhs.shape == rhs.shape, lhs.isContiguous, rhs.isContiguous {
            let result = vDSP.multiply(lhs.storage.data, rhs.storage.data)
            return Self(lhs.shape, result)
        }
        return performOperationSlow(lhs, rhs, *)
    }
}

