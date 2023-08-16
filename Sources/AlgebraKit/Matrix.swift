import Foundation
import Accelerate

public struct Matrix {
    public private(set) var storage: [Float]
    public private(set) var rows: Int
    public private(set) var cols: Int

    public init(_ data: [Float]) {
        self.rows = 1
        self.cols = data.count
        self.storage = data
    }

    private func indexIsValid(row: Int, col: Int) -> Bool {
        row >= 0 && row < rows && col >= 0 && col < cols
    }

    public subscript(row: Int, col: Int) -> Float {
        get {
            assert(indexIsValid(row: row, col: col), "Index out of range")
            return storage[(row * cols) + col]
        }
        set {
            assert(indexIsValid(row: row, col: col), "Index out of range")
            storage[(row * cols) + col] = newValue
        }
    }

    public init(rows: Int, cols: Int, data: [Float]) {
        assert(data.count == rows * cols)
        self.rows = rows
        self.cols = cols
        self.storage = data
    }

    public init(rows: Int, cols: Int, repeating constant: Float = 0.0) {
        self.rows = rows
        self.cols = cols
        self.storage = Array(repeating: constant, count: rows * cols)
    }

    public init(as other: Matrix, repeating constant: Float = 0.0) {
        self.rows = other.rows
        self.cols = other.cols
        self.storage = Array(repeating: constant, count: rows * cols)
    }

    public init(as other: Matrix, data: [Float]) {
        assert(data.count == other.rows * other.cols)
        self.rows = other.rows
        self.cols = other.cols
        self.storage = data
    }
}

extension Matrix {
    public static func random(rows: Int, cols: Int) -> Matrix {
        Matrix(rows: rows, cols: cols, data: (0..<(rows*cols)).map { _ in Float.random(in: -0.1...0.1) })
    }

    public func transposed() -> Matrix {
        Matrix.transpose(self)
    }

    public static func transpose(_ m: Matrix) -> Matrix {
        let resultSize = m.rows * m.cols
        let result = UnsafeMutablePointer<Float>.allocate(capacity: resultSize)
        defer { result.deallocate() }
        m.storage.withUnsafeBufferPointer { mPtr in
            vDSP_mtrans(
                mPtr.baseAddress!,
                1,
                result,
                1,
                vDSP_Length(m.rows),
                vDSP_Length(m.cols)
            )
        }
        return Matrix(
            rows: m.cols,
            cols: m.rows,
            data: Array(UnsafeBufferPointer(start: result, count: resultSize))
        )
    }

    public static func *(lhs: Matrix, rhs: Matrix) -> Matrix {
        matmul(m1: lhs, m2: rhs)
    }

    public static func +(lhs: Matrix, rhs: Matrix) -> Matrix {
        assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        return Matrix(rows: lhs.rows, cols: lhs.cols, data: Array(vDSP.add(lhs.storage, rhs.storage)))
    }

    public static func +(lhs: Matrix, rhs: Float) -> Matrix {
        Matrix(rows: lhs.rows, cols: lhs.cols, data: Array(vDSP.add(rhs, lhs.storage)))
    }

    public static func +(lhs: Float, rhs: Matrix) -> Matrix {
        rhs + lhs
    }

    public static func -(lhs: Matrix, rhs: Float) -> Matrix {
        lhs + (-rhs)
    }

    public static func *(lhs: Float, rhs: Matrix) -> Matrix {
        Matrix(rows: rhs.rows, cols: rhs.cols, data: Array(vDSP.multiply(lhs, rhs.storage)))
    }

    public static func *(lhs: Matrix, rhs: Float) -> Matrix {
        rhs * lhs
    }

    public static func /(lhs: Matrix, rhs: Float) -> Matrix {
        lhs * (1.0 / rhs)
    }

    public static func -(lhs: Matrix, rhs: Matrix) -> Matrix {
        assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        return Matrix(rows: rhs.rows, cols: rhs.cols, data: Array(vDSP.subtract(lhs.storage, rhs.storage)))
    }

    public static func elementwiseMul(
        m1: Matrix,
        m2: Matrix
    ) -> Matrix {
        assert(m1.rows == m2.rows && m1.cols == m2.cols)
        var result = Array<Float>(repeating: 0, count: m1.rows * m1.cols)
        vDSP.multiply(m1.storage, m2.storage, result: &result)
        return Matrix(as: m1, data: Array(result))
    }

    public static func matmul(
        m1: Matrix,
        m2: Matrix
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
}

extension Matrix: CustomDebugStringConvertible {
    private static var valueFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.decimalSeparator = "."
        formatter.minimumIntegerDigits = 2
        formatter.maximumIntegerDigits = 2
        formatter.minimumFractionDigits = 2
        formatter.maximumFractionDigits = 2
        return formatter
    }()

    public var debugDescription: String {
        var result = ""
        result += "[rows: \(rows), cols: \(cols)]\n"
        for r in 0..<rows {
            var rowNums: [Float] = []
            for c in 0..<cols {
                rowNums.append(storage[r * cols + c])
            }
            result += "[\(rowNums.map({ Matrix.valueFormatter.string(from: NSNumber(value: $0)) ?? "" }).joined(separator: ", "))]\n"
        }
        return result
    }
}
