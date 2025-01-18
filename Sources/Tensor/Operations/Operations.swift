extension Tensor {

    //
    //    public static func +(lhs: Matrix, rhs: Float) -> Matrix {
    //        Matrix(size: lhs.size, data: Array(vDSP.add(rhs, lhs.storage)))
    //    }
    //
    //    public static func +(lhs: Float, rhs: Matrix) -> Matrix {
    //        rhs + lhs
    //    }
    //
    //    public static func +=(lhs: inout Matrix, rhs: MatrixConvertible) {
    //        let rhs = rhs.asMatrix()
    //        return lhs = lhs + rhs
    //    }
    //
    //    public static func -=(lhs: inout Matrix, rhs: MatrixConvertible) {
    //        let rhs = rhs.asMatrix()
    //        return lhs = lhs - rhs
    //    }
    //
    //    public static func -(lhs: Matrix, rhs: Float) -> Matrix {
    //        lhs + (-rhs)
    //    }
    //
    //    public static func *(lhs: Float, rhs: Matrix) -> Matrix {
    //        Matrix(size: rhs.size, data: Array(vDSP.multiply(lhs, rhs.storage)))
    //    }
    //
    //    public static func *(lhs: Matrix, rhs: Float) -> Matrix {
    //        rhs * lhs
    //    }
    //
    //    public static func /(lhs: Matrix, rhs: Float) -> Matrix {
    //        lhs * (1.0 / rhs)
    //    }
    //

    public static func *(lhs: Self, rhs: Self) -> Self {
        performOperation(lhs, rhs, *)
    }

    public static func /(lhs: Self, rhs: Self) -> Self {
        performOperation(lhs, rhs, /)
    }

    public static func +(lhs: Self, rhs: Self) -> Self {
        performOperation(lhs, rhs, +)
    }

    public static func -(lhs: Self, rhs: Self) -> Self {
        performOperation(lhs, rhs, -)
    }

    static func performOperation(_ lhs: Self, _ rhs: Self, _ operation: (Double, Double) -> Double) -> Self {
        guard let resultShape = Tensor.broadcastShapes(lhs.shape, rhs.shape) else {
            fatalError("Shapes doesn't match and can't be broadcasted")
        }
        let lhs = lhs.broadcastTo(resultShape)!
        let rhs = rhs.broadcastTo(resultShape)!
        var result = Tensor.init(zeros: resultShape)
        result.forEachIndex {
            result.assign(operation(lhs[$0], rhs[$0]), at: $0)
        }
        return result
    }
}
