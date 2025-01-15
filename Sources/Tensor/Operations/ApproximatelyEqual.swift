import Numerics

extension Tensor {
    public func isApproximatelyEqual(to other: Tensor<Element>) -> Bool {
        guard shape == other.shape else {
            return false
        }
        var result = true
        forEachIndex { index in
            guard self[index].isApproximatelyEqual(to: other[index]) else {
                result = false
                return
            }
        }
        return result
    }
}
