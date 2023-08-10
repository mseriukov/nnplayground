import Foundation
import Accelerate

public struct Matrix {
    public private(set) var storage: ContiguousArray<Float>
    public private(set) var rows: Int
    public private(set) var cols: Int

    public init(rows: Int, cols: Int, data: ContiguousArray<Float>) {
        assert(data.count == rows * cols)
        self.rows = rows
        self.cols = cols
        self.storage = data
    }

    public init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        self.storage = ContiguousArray(repeating: 0.0, count: rows * cols)
    }
}

extension Matrix {
    public static func *(lhs: Matrix, rhs: Matrix) -> Matrix {
        matmul(m1: lhs, m2: rhs)
    }

    static func matmul(m1: Matrix, m2: Matrix) -> Matrix {
        let resultSize = m1.rows * m2.cols
        let result = UnsafeMutablePointer<Float>.allocate(capacity: resultSize)
        m1.storage.withUnsafeBufferPointer { m1ptr in
            m2.storage.withUnsafeBufferPointer { m2ptr in
                cblas_sgemm(
                    CblasRowMajor,      // Row or column majir
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
                    2.0,                // Scaling factor.
                    result,
                    Int32(m1.rows)
                )
            }
        }
        return Matrix(
            rows: m1.rows,
            cols: m2.cols,
            data: ContiguousArray(UnsafeBufferPointer(start: result, count: resultSize))
        )
    }
}


