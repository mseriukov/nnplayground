extension Tensor {
    public func matmul(_ other: Self) -> Self {
        precondition(shape.count == 2 && other.shape.count == 2, "Both tensors must be 2D for matmul.")
        precondition(shape[1] == other.shape[0], "Inner dimensions must match for matmul.")

        let m = shape[0]
        let n = shape[1]
        let p = other.shape[1]

        let result = Self(zeros: [m, p])
        result.forEachIndex { resultIndex in
            let i = resultIndex[0]
            let j = resultIndex[1]

            var sum: Element = 0.0
            for k in 0..<n {
                let a = self.flatIndex([i, k])
                let b = other.flatIndex([k, j])
                sum += self.storage[a] * other.storage[b]
            }
            result.storage[result.flatIndex(resultIndex)] = sum
        }
        return result
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
            result.assign(start: [i, 0, 0], tensor: c.unsqueeze(axis: 0))
        }

        return result
    }
}
