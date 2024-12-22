import Foundation

public func im2col(
    _ input: Matrix,
    _ filterSize: Int
) -> Matrix {
    var result = Matrix(rows: filterSize * filterSize, cols: (input.rows - filterSize + 1) * (input.cols - filterSize + 1))
    for r in 0..<(input.rows - filterSize + 1) {
        for c in 0..<(input.cols - filterSize + 1) {
            for sr in 0..<filterSize {
                for sc in 0..<filterSize {
                    result[sr * filterSize + sc, r * (input.cols - filterSize + 1) + c] = input[r + sr, c + sc]
                }
            }
        }
    }
    return result
}

public func im2col(
    input: Matrix,
    filterSize: Size,
    aStride: Stride,
    padding: Padding,
    dilation: Dilation
) -> Matrix {
    let dilatedFilterWidth = dilation.horizontal * (filterSize.cols - 1) + 1
    let dilatedFilterHeight = dilation.vertical * (filterSize.rows - 1) + 1
    let resultCols = Int(floor(Float(padding.left + padding.right + input.cols - dilatedFilterWidth) / Float(aStride.horizontal) + 1.0))
    let resultRows = Int(floor(Float(padding.top + padding.bottom + input.rows - dilatedFilterHeight) / Float(aStride.vertical) + 1.0))
    var result = Matrix(rows: filterSize.rows * filterSize.cols, cols: resultRows * resultCols)

    for r in stride(from: 0, to: (input.rows - dilatedFilterHeight + 1), by: aStride.vertical) {
        for c in stride(from: 0, to: (input.cols - dilatedFilterWidth + 1), by: aStride.horizontal) {
            for sr in 0..<filterSize.rows {
                for sc in 0..<filterSize.cols {
                    result[
                        sr * filterSize.cols + sc,
                        r * (input.cols - dilatedFilterWidth + 1) + c
                    ] = input[
                        r + sr * dilation.vertical,
                        c + sc * dilation.horizontal
                    ]
                }
            }
        }
    }
    return result
}
