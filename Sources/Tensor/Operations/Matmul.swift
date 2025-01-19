import Accelerate

extension Tensor {
    public func matmul(_ other: Self) -> Self {
        let t1 = makeContiguous()
        let t2 = other.makeContiguous()

        let m = t1.shape[0]
        let n = t1.shape[1]
        let p = t2.shape[1]

        let resultSize =  m * p

        let result = UnsafeMutablePointer<Element>.allocate(capacity: resultSize)
        defer { result.deallocate() }
        t1.storage.data.withUnsafeBufferPointer { t1ptr in
            t2.storage.data.withUnsafeBufferPointer { t2ptr in
                cblas_sgemm(
                    CblasRowMajor,      // Row or column major
                    CblasNoTrans,       // Should transpose t1
                    CblasNoTrans,       // Should transpose mt2
                    Int32(m),
                    Int32(p),
                    Int32(n),
                    1.0,                // Scaling factor
                    t1ptr.baseAddress,
                    Int32(n),
                    t2ptr.baseAddress,
                    Int32(p),
                    0.0,                // Scaling factor.
                    result,
                    Int32(p)
                )
            }
        }
        return Self([m, p], Array(UnsafeBufferPointer(start: result, count: resultSize)))
    }

    public static func batchedMatMul(_ A: Self, _ B: Self) -> Self {
        assert(A.shape.count == 3 && B.shape.count == 3, "Both tensors must be 3D.")
        assert(A.shape[0] == B.shape[0], "Batch sizes must match.")
        assert(A.shape[2] == B.shape[1], "Inner dimensions must match.")

        let batchSize = A.shape[0]
        let m = A.shape[1]
        let n = A.shape[2]
        let p = B.shape[2]

        var result = Tensor(zeros: [batchSize, m, p])

        for i in 0..<batchSize {
            let a = A.slice(start: [i, 0, 0], shape: [1, m, n]).squeezed([0])
            let b = B.slice(start: [i, 0, 0], shape: [1, n, p]).squeezed([0])
            let c = a.matmul(b)
            result.assign(start: [i, 0, 0], tensor: c.unsqueezed(axis: 0))
        }

        return result
    }
}
