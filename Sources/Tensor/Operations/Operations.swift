import Accelerate

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
    
    public static func /(lhs: Self, rhs: Self) -> Self {
        performOperationSlow(lhs, rhs, /)
    }

    public static func -(lhs: Self, rhs: Self) -> Self {
        if lhs.shape == rhs.shape, lhs.isContiguous, rhs.isContiguous {
            let result = vDSP.subtract(lhs.dataSlice, rhs.dataSlice)
            return Self(lhs.shape, result)
        }
        return performOperationSlow(lhs, rhs, -)
    }
}
