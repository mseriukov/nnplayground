import Accelerate

extension Tensor {
    public mutating func normalize() {
        var mean: Float = 0
        var stdDev: Float = 0
        vDSP_normalize(
            storage.buffer.baseAddress!,
            1,
            storage.buffer.baseAddress!,
            1,
            &mean,
            &stdDev,
            vDSP_Length(storage.buffer.count)
        )
    }

    public func normalized() -> Self {
        var copy = copy()
        copy.normalize()
        return copy
    }
}
