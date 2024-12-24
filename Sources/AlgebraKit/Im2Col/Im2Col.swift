import Foundation

public func im2col(
    _ input: Matrix,
    _ filterSize: Int
) -> Matrix {
    var result = Matrix(
        size: Size(
            filterSize * filterSize,
            (input.size.rows - filterSize + 1) * (input.size.cols - filterSize + 1)
        )
    )
    for r in 0..<(input.size.rows - filterSize + 1) {
        for c in 0..<(input.size.cols - filterSize + 1) {
            for sr in 0..<filterSize {
                for sc in 0..<filterSize {
                    result[sr * filterSize + sc, r * (input.size.cols - filterSize + 1) + c] = input[r + sr, c + sc]
                }
            }
        }
    }
    return result
}

public func im2col(
    input: Matrix,
    filter: FilterDescriptor,
    aStride: Stride,
    padding: Padding
) -> Matrix {
    let resultCols = Int(floor(Float(padding.left + padding.right + input.size.cols - filter.effectiveSize.cols) / Float(aStride.horizontal) + 1.0))
    let resultRows = Int(floor(Float(padding.top + padding.bottom + input.size.rows - filter.effectiveSize.rows) / Float(aStride.vertical) + 1.0))
    var result = Matrix(
        size: Size(filter.size.rows * filter.size.cols, resultRows * resultCols))

    for r in stride(from: 0, to: (input.size.rows - filter.effectiveSize.rows + 1), by: aStride.vertical) {
        for c in stride(from: 0, to: (input.size.cols - filter.effectiveSize.cols + 1), by: aStride.horizontal) {
            for sr in 0..<filter.size.rows {
                for sc in 0..<filter.size.cols {
                    result[
                        sr * filter.size.cols + sc,
                        r * (input.size.cols - filter.effectiveSize.cols + 1) + c
                    ] = input[
                        r + sr * filter.dilation.vertical,
                        c + sc * filter.dilation.horizontal
                    ]
                }
            }
        }
    }
    return result
}
