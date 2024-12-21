//
//  Im2Col.swift
//  nnplayground
//
//  Created by Mikhail Seriukov on 21/12/2024.
//


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
