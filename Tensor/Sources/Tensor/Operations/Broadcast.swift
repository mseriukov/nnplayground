extension Tensor {
    static func broadcastShapes(_ shape1: [Int], _ shape2: [Int]) -> [Int]? {
        let maxRank = Swift.max(shape1.count, shape2.count)
        var resultShape = [Int](repeating: 1, count: maxRank)

        for i in 0..<maxRank {
            let dim1 = i < maxRank - shape1.count ? 1 : shape1[i - (maxRank - shape1.count)]
            let dim2 = i < maxRank - shape2.count ? 1 : shape2[i - (maxRank - shape2.count)]

            if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
                resultShape[i] = Swift.max(dim1, dim2)
            } else {
                return nil
            }
        }
        return resultShape
    }

    public func broadcastTo(_ shape: [Int]) -> Self? {
        if isScalar {
            var scalarView = self
            scalarView.shape = shape
            scalarView.strides = Array(repeating: 0, count: shape.count)
            return scalarView
        }

        guard let newShape = Tensor.broadcastShapes(self.shape, shape) else {
            return nil
        }

        var newStrides = [Int](repeating: 0, count: newShape.count)
        let originalStrides = self.strides
        let originalShape = self.shape

        let strideOffset = newShape.count - originalShape.count

        for i in 0..<newShape.count {
            if i < strideOffset {
                newStrides[i] = 0
            } else if originalShape[i - strideOffset] == newShape[i] {
                newStrides[i] = originalStrides[i - strideOffset]
            } else if originalShape[i - strideOffset] == 1 {
                newStrides[i] = 0
            } else {
                return nil
            }
        }

        var broadcastedTensor = self
        broadcastedTensor.shape = newShape
        broadcastedTensor.strides = newStrides
        return broadcastedTensor
    }
}
