import Foundation
import Accelerate

public enum RandomKind {
    // Normal distribution in [0...1]
    case normal

    case kaiming(inputChannels: Int)
}

public protocol MatrixConvertible {
    func asMatrix() -> Matrix
}

public struct Matrix {
    public private(set) var storage: [Float]
    public private(set) var rows: Int
    public private(set) var cols: Int

    public init(_ data: [Float]) {
        self.rows = 1
        self.cols = data.count
        self.storage = data
    }

    public static var zero: Matrix {
        Matrix(rows: 0, cols: 0, data: [])
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

    public static func identity(size: Int) -> Matrix {
        var m = Matrix(rows: size, cols: size, repeating: 0.0)
        for i in 0..<size {
            m[i, i] = 1.0
        }
        return m
    }

    public static func diagonal(from im: Matrix) -> Matrix {
        assert(im.rows == 1)
        let size = im.cols
        var m = Matrix(rows: size, cols: size, repeating: 0.0)
        for i in 0..<size {
            m[i, i] = im[0, i]
        }
        return m
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

// MARK: - Random
extension Matrix {
    public static func random(as m: Matrix, kind: RandomKind, seed: UInt32?) -> Matrix {
        random(rows: m.rows, cols: m.cols, kind: kind, seed: seed)
    }
    
    public static func random(rows: Int, cols: Int, kind: RandomKind, seed: UInt32?) -> Matrix {
        var result = Matrix(
            rows: rows,
            cols: cols,
            data: Array(count: rows * cols, mean: 0, std: 1, seed: seed ?? 1)
        )

        switch kind {
        case .normal:
            break

        case let .kaiming(inputChannels):
            let variance = 2.0 / Float(inputChannels)
            let scale = sqrt(variance)
            result.mapInPlace { $0 *= scale }
        }

        return result
    }
}

extension Matrix {
    public func padded(_ padding: Padding, value: Float = 0.0) -> Matrix {
        let oldRows = rows
        let oldCols = cols
        let newRows = padding.top + oldRows + padding.bottom
        let newCols = padding.left + oldCols + padding.right

        var result = Array<Float>(repeating: value, count: newRows * newCols)
        for r in 0..<oldRows {
            for c in 0..<oldCols {
                result[(r + padding.top) * newCols + (c + padding.left)] = storage[r * oldCols + c]
            }
        }
        return Matrix(rows: newRows, cols: newCols, data: result)
    }

    public mutating func pad(_ padding: Padding, value: Float = 0.0) {
        self = padded(padding, value: value)
    }

    public mutating func reshape(rows: Int, cols: Int) {
        precondition(self.storage.count == rows * cols, "Size doesn't match")
        self.rows = rows
        self.cols = cols
    }

    public mutating func normalize() {
        storage.normalize()
    }

    public mutating func invert() {
        storage.invert()
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

    public static func *(lhs: Matrix, rhs: MatrixConvertible) -> Matrix {
        let rhs = rhs.asMatrix()
        return matmul(lhs, rhs)
    }

    public static func +(lhs: Matrix, rhs: MatrixConvertible) -> Matrix {
        let rhs = rhs.asMatrix()
        assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        return Matrix(rows: lhs.rows, cols: lhs.cols, data: Array(vDSP.add(lhs.storage, rhs.storage)))
    }

    public static func +(lhs: Matrix, rhs: Float) -> Matrix {
        Matrix(rows: lhs.rows, cols: lhs.cols, data: Array(vDSP.add(rhs, lhs.storage)))
    }

    public static func +(lhs: Float, rhs: Matrix) -> Matrix {
        rhs + lhs
    }

    public static func +=(lhs: inout Matrix, rhs: MatrixConvertible) {
        let rhs = rhs.asMatrix()
        return lhs = lhs + rhs
    }

    public static func -=(lhs: inout Matrix, rhs: MatrixConvertible) {
        let rhs = rhs.asMatrix()
        return lhs = lhs - rhs
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

    public static func -(lhs: Matrix, rhs: MatrixConvertible) -> Matrix {
        let rhs = rhs.asMatrix()
        assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols)
        return Matrix(rows: rhs.rows, cols: rhs.cols, data: Array(vDSP.subtract(lhs.storage, rhs.storage)))
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

extension Matrix: Equatable { }

extension Matrix {
    mutating func mapInPlace(_ transform: (inout Float) -> Void) {
        storage.mapInPlace(transform)
    }
}

extension Matrix: MatrixConvertible {
    public func asMatrix() -> Matrix { self }
}
