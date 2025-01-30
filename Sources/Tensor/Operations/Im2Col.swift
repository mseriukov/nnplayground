extension Tensor {
    public func im2col(
        kernelSize: (Int, Int),
        stride: (Int, Int) = (1, 1),
        padding: (Int, Int) = (0, 0),
        dilation: (Int, Int) = (1, 1)
    ) throws -> Tensor {
        let (batchSize, inChannels, inHeight, inWidth) = (shape[0], shape[1], shape[2], shape[3])
        let (kernelH, kernelW) = kernelSize
        let (strideH, strideW) = stride
        let (padH, padW) = padding
        let (dilationH, dilationW) = dilation

        let outHeight = (inHeight + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1
        let outWidth = (inWidth + 2 * padW - dilationW * (kernelW - 1) - 1) / strideW + 1

        var output = Tensor(shape: [batchSize, inChannels, kernelH, kernelW, outHeight, outWidth], value: 0)

        for b in 0..<batchSize {
            for c in 0..<inChannels {
                for kh in 0..<kernelH {
                    for kw in 0..<kernelW {
                        let hStart = kh * dilationH - padH
                        let wStart = kw * dilationW - padW

                        for oh in 0..<outHeight {
                            for ow in 0..<outWidth {
                                let h = hStart + oh * strideH
                                let w = wStart + ow * strideW

                                if h >= 0, h < inHeight, w >= 0, w < inWidth {
                                    output[b, c, kh, kw, oh, ow] = self[b, c, h, w]
                                } else {
                                    output[b, c, kh, kw, oh, ow] = 0 // Zero padding
                                }
                            }
                        }
                    }
                }
            }
        }

        return try output.reshape([batchSize, inChannels * kernelH * kernelW, outHeight * outWidth])
    }
}
