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
    public private(set) var size: Size

    public init(size: Size, data: [Float]) {
        assert(data.count == size.elementCount)
        self.size = size
        self.storage = data
    }

    public init(size: Size, repeating constant: Float = 0.0) {
        self.init(
            size: size,
            data: Array(repeating: constant, count: size.elementCount)
        )
    }

    public init(as other: Matrix, repeating constant: Float = 0.0) {
        self.init(
            size: other.size,
            data: Array(repeating: constant, count: other.size.elementCount)
        )
    }

    public init(as other: Matrix, data: [Float]) {
        self.init(
            size: other.size,
            data: data
        )
    }

    public init(_ data: [Float]) {
        self.init(
            size: Size(1, data.count),
            data: data
        )
    }

    public subscript(row: Int, col: Int) -> Float {
        get {
            assert(indexIsValid(row: row, col: col), "Index out of range")
            return storage[(row * size.cols) + col]
        }
        set {
            assert(indexIsValid(row: row, col: col), "Index out of range")
            storage[(row * size.cols) + col] = newValue
        }
    }

    private func indexIsValid(row: Int, col: Int) -> Bool {
        row >= 0 && row < size.rows && col >= 0 && col < size.cols
    }
}

extension Matrix {
    public static var zero: Matrix {
        Matrix(size: 0, data: [])
    }

    public static func identity(size: Int) -> Matrix {
        var m = Matrix(size: Size(size), repeating: 0.0)
        for i in 0..<size {
            m[i, i] = 1.0
        }
        return m
    }

    public static func diagonal(from im: Matrix) -> Matrix {
        assert(im.size.rows == 1)
        var m = Matrix(size: Size(im.size.cols), repeating: 0.0)
        for i in 0..<im.size.cols {
            m[i, i] = im[0, i]
        }
        return m
    }
}

// MARK: - Random
extension Matrix {
    public static func random(as m: Matrix, kind: RandomKind, seed: UInt32?) -> Matrix {
        random(size: m.size, kind: kind, seed: seed)
    }
    
    public static func random(size: Size, kind: RandomKind, seed: UInt32?) -> Matrix {
        var result = Matrix(
            size: size,
            data: Array(count: size.elementCount, mean: 0, std: 1, seed: seed ?? 1)
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
        let oldRows = size.rows
        let oldCols = size.cols
        let newRows = padding.top + oldRows + padding.bottom
        let newCols = padding.left + oldCols + padding.right

        var result = Array<Float>(repeating: value, count: newRows * newCols)
        for r in 0..<oldRows {
            for c in 0..<oldCols {
                result[(r + padding.top) * newCols + (c + padding.left)] = storage[r * oldCols + c]
            }
        }
        return Matrix(size: Size(newRows, newCols), data: result)
    }

    public mutating func pad(_ padding: Padding, value: Float = 0.0) {
        self = padded(padding, value: value)
    }

    public mutating func reshape(size: Size) {
        precondition(self.storage.count == size.elementCount, "Size doesn't match")
        self.size = size
    }

    public mutating func scaleToUnitInterval() {
        storage.scaleToUnitInterval()
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
        let resultSize = m.size.elementCount
        let result = UnsafeMutablePointer<Float>.allocate(capacity: resultSize)
        defer { result.deallocate() }
        m.storage.withUnsafeBufferPointer { mPtr in
            vDSP_mtrans(
                mPtr.baseAddress!,
                1,
                result,
                1,
                vDSP_Length(m.size.rows),
                vDSP_Length(m.size.cols)
            )
        }
        return Matrix(
            size: Size(m.size.cols, m.size.rows),
            data: Array(UnsafeBufferPointer(start: result, count: resultSize))
        )
    }

    public static func *(lhs: Matrix, rhs: MatrixConvertible) -> Matrix {
        let rhs = rhs.asMatrix()
        return matmul(lhs, rhs)
    }

    public static func +(lhs: Matrix, rhs: MatrixConvertible) -> Matrix {
        let rhs = rhs.asMatrix()
        assert(lhs.size.rows == rhs.size.rows && lhs.size.cols == rhs.size.cols)
        return Matrix(size: lhs.size, data: Array(vDSP.add(lhs.storage, rhs.storage)))
    }

    public static func +(lhs: Matrix, rhs: Float) -> Matrix {
        Matrix(size: lhs.size, data: Array(vDSP.add(rhs, lhs.storage)))
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
        Matrix(size: rhs.size, data: Array(vDSP.multiply(lhs, rhs.storage)))
    }

    public static func *(lhs: Matrix, rhs: Float) -> Matrix {
        rhs * lhs
    }

    public static func /(lhs: Matrix, rhs: Float) -> Matrix {
        lhs * (1.0 / rhs)
    }

    public static func -(lhs: Matrix, rhs: MatrixConvertible) -> Matrix {
        let rhs = rhs.asMatrix()
        assert(lhs.size.rows == rhs.size.rows && lhs.size.cols == rhs.size.cols)
        return Matrix(size: rhs.size, data: Array(vDSP.subtract(lhs.storage, rhs.storage)))
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
        result += "[rows: \(size.rows), cols: \(size.cols)]\n"
        for r in 0..<size.rows {
            var rowNums: [Float] = []
            for c in 0..<size.cols {
                rowNums.append(storage[r * size.cols + c])
            }
            result += "[\(rowNums.map({ Matrix.valueFormatter.string(from: NSNumber(value: $0)) ?? "" }).joined(separator: ", "))]\n"
        }
        return result
    }
}

extension Matrix: Equatable { }

extension Matrix {
    public mutating func mapInPlace(_ transform: (inout Float) -> Void) {
        storage.mapInPlace(transform)
    }

    public func map(_ transform: (Float) -> Float) -> Matrix {
        Matrix(size: size , data: storage.map(transform))
    }
}

extension Matrix: MatrixConvertible {
    public func asMatrix() -> Matrix { self }
}
