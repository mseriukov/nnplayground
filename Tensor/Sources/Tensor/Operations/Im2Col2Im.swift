extension Tensor {
    public func im2col(
        kernelSize: (Int, Int),
        stride: (Int, Int) = (1, 1),
        padding: (Int, Int) = (0, 0),
        dilation: (Int, Int) = (1, 1)
    ) throws -> Tensor {
        let (batchSize, inputChannels, inputRows, inputCols) = (shape[0], shape[1], shape[2], shape[3])
        let (kernelRows, kernelCols) = kernelSize
        let (strideRows, strideCols) = stride
        let (paddingRows, paddingCols) = padding
        let (dilationRows, dilationCols) = dilation

        let outputRows = (inputRows + 2 * paddingRows - dilationRows * (kernelRows - 1) - 1) / strideRows + 1
        let outputCols = (inputCols + 2 * paddingCols - dilationCols * (kernelCols - 1) - 1) / strideCols + 1

        var output = Tensor(shape: [batchSize, inputChannels, kernelRows, kernelCols, outputRows, outputCols], value: 0)

        for b in 0..<batchSize {
            for c in 0..<inputChannels {
                for kr in 0..<kernelRows {
                    for kc in 0..<kernelCols {
                        let hStart = kr * dilationRows - paddingRows
                        let wStart = kc * dilationCols - paddingCols

                        for or in 0..<outputRows {
                            for oc in 0..<outputCols {
                                let row = hStart + or * strideRows
                                let col = wStart + oc * strideCols

                                if row >= 0, row < inputRows, col >= 0, col < inputCols {
                                    output[b, c, kr, kc, or, oc] = self[b, c, row, col]
                                } else {
                                    output[b, c, kr, kc, or, oc] = 0 // Zero padding
                                }
                            }
                        }
                    }
                }
            }
        }
        return try output.reshape([batchSize, inputChannels * kernelRows * kernelCols, outputRows * outputCols])
    }

    public func col2im(
        inputShape: [Int],
        kernelSize: (Int, Int),
        stride: (Int, Int),
        padding: (Int, Int),
        dilation: (Int, Int)
    ) -> Tensor {
        let (batchSize, channels, inputRows, inputCols) = (inputShape[0], inputShape[1], inputShape[2], inputShape[3])
        let (kernelRows, kernelCols) = kernelSize
        let (strideRows, strideCols) = stride
        let (paddingRows, paddingCols) = padding
        let (dilationRows, dilationCols) = dilation

        let outputRows = (inputRows + 2 * paddingRows - (kernelRows - 1) * dilationRows - 1) / strideRows + 1
        let outputCols = (inputCols + 2 * paddingCols - (kernelCols - 1) * dilationCols - 1) / strideCols + 1

        var result = Tensor(zeros: [batchSize, channels, inputRows, inputCols])

        var colIndex = 0
        for b in 0..<batchSize {
            for c in 0..<channels {
                for kr in 0..<kernelRows {
                    for kc in 0..<kernelCols {
                        let offsetRows = kr * dilationRows
                        let offsetCols = kc * dilationCols
                        for or in 0..<outputRows {
                            let ir = or * strideRows - paddingRows + offsetRows
                            for oc in 0..<outputCols {
                                let ic = oc * strideCols - paddingCols + offsetCols
                                if ir >= 0, ir < inputRows, ic >= 0, ic < inputCols {
                                    result[b, c, ir, ic] += self[b, colIndex, or * outputCols + oc]
                                }
                            }
                        }
                        colIndex += 1
                    }
                }
            }
        }
        return result
    }
}
