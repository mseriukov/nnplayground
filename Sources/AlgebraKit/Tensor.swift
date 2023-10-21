import Foundation
import Accelerate

// tensor: <N, C, H, W>
// index: <n, c, h, w>
// strides: <CHW, HW, W, 1>
// offset(n,c,h,w) = stride_n * n + stride_c * c + stride_h * h + stride_w * w = CHW * n + HW * c + W * h + 1 * w


public struct Tensor {
    public private(set) var storage: [Float]
    public private(set) var shape: [Int]
    public private(set) var strides: [Int] = []

    public init(_ data: [Float], shape: [Int]) {
        assert(data.count == shape.reduce(1, *), "Input data doesn't match provided tensor shape.")
        self.shape = shape
        self.storage = data

        strides.append(1)
        for shapeEntry in shape.dropFirst().reversed() {
            strides.insert(strides.first! * shapeEntry, at: 0)
        }

    }
}
