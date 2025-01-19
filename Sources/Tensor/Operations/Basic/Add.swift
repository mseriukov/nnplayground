import Accelerate

extension Tensor {
    public func incremented(by scalar: Element) -> Self {
        let t = makeContiguous()
        let result = vDSP.add(scalar, t.storage.data)
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
            let result = vDSP.multiply(addition: (lhs.storage.data, rhs.storage.data), 1)
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
