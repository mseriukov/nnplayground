import Accelerate

extension Tensor {
    public func transposed() -> Self {
        precondition(shape.count == 2, "Must be 2D for now.")
        let t = makeContiguous()
        let result = UnsafeMutablePointer<Double>.allocate(capacity: size)
        defer { result.deallocate() }
        t.storage.data.withUnsafeBufferPointer { mPtr in
            vDSP_mtransD(
                mPtr.baseAddress!,
                1,
                result,
                1,
                vDSP_Length(t.shape[1]),
                vDSP_Length(t.shape[0])
            )
        }
        return Self(shape.reversed(), Array(UnsafeBufferPointer(start: result, count: size)))
    }
}
