import Foundation
import Accelerate

// tensor: <N, C, H, W>
// index: <n, c, h, w>
// strides: <CHW, HW, W, 1>
// offset(n,c,h,w) = stride_n * n + stride_c * c + stride_h * h + stride_w * w = CHW * n + HW * c + W * h + 1 * w


public struct Tensor {

    public enum Errors: Error {
        case shapeIsZeroLength
        case broadcastIsntPossible
    }

    public private(set) var storage: [Float]
    public private(set) var shape: [Int]
    public private(set) var strides: [Int] = []

    public init(_ data: [Float], shape: [Int]) {
        assert(data.count == shape.reduce(1, *), "Input data doesn't match provided tensor shape.")
        self.shape = shape
        self.storage = data
        reshape(shape)
    }

    mutating func reshape(_ shape: [Int]) {
        guard shape.reduce(1, *) == storage.count else { fatalError("New shape doesn't fit the data.") }
        strides = [1]
        for shapeEntry in shape.dropFirst().reversed() {
            strides.insert(strides.first! * shapeEntry, at: 0)
        }
    }

    func element(at indicies: [Int]) -> Float {
        guard indicies.count == shape.count else { fatalError("indicies vs shape mismatch") }
        var offset: Int = 0
        for i in 0..<shape.count {
            offset += shape[i] * indicies[i]
        }
        return storage[offset]
    }

    private func broadcast(_ shape1: [Int], _ shape2: [Int]) throws -> [Int] {
        guard shape1.count > 0, shape2.count > 0 else {
            throw Errors.shapeIsZeroLength
        }
        let N1 = shape1.count
        let N2 = shape2.count
        let N = max(N1, N2)

        var result = Array(repeating: 1, count: N)

        for i in (0..<N).reversed() {
            let n1 = N1 - N + i
            let n2 = N2 - N + i
            let d1 = n1 >= 0 ? shape1[n1] : 1
            let d2 = n2 >= 0 ? shape2[n2] : 1

            if d1 == 1 {
                result[i] = d2
            } else if d2 == 1 {
                result[i] = d1
            } else if d1 == d2 {
                result[i] = d1
            } else {
                throw Errors.broadcastIsntPossible
            }
        }
        return result
    }
}
