import Accelerate

public func exp(_ input: Matrix) -> Matrix {
    Matrix(as: input, data: input.storage.map { exp($0) })
}

public func max(_ input: Matrix) -> Float {
    input.storage.max()!
}

public func elementwiseMul(
    _ m1: Matrix,
    _ m2: Matrix
) -> Matrix {
    assert(m1.rows == m2.rows && m1.cols == m2.cols)
    var result = Array<Float>(repeating: 0, count: m1.rows * m1.cols)
    vDSP.multiply(m1.storage, m2.storage, result: &result)
    return Matrix(as: m1, data: Array(result))
}

public func matmul(
    _ m1: Matrix,
    _ m2: Matrix
) -> Matrix {
    let resultSize = m1.rows * m2.cols
    let result = UnsafeMutablePointer<Float>.allocate(capacity: resultSize)
    defer { result.deallocate() }
    m1.storage.withUnsafeBufferPointer { m1ptr in
        m2.storage.withUnsafeBufferPointer { m2ptr in
            cblas_sgemm(
                CblasRowMajor,      // Row or column major
                CblasNoTrans,       // Should transpose m1
                CblasNoTrans,       // Should transpose m2
                Int32(m1.rows),
                Int32(m2.cols),
                Int32(m1.cols),
                1.0,                // Scaling factor
                m1ptr.baseAddress,
                Int32(m1.cols),
                m2ptr.baseAddress,
                Int32(m2.cols),
                0.0,                // Scaling factor.
                result,
                Int32(m2.cols)
            )
        }
    }
    return Matrix(
        rows: m1.rows,
        cols: m2.cols,
        data: Array(UnsafeBufferPointer(start: result, count: resultSize))
    )
}
