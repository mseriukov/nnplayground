import Accelerate

public struct NDArray {
    public private(set) var storage: [Double]
    public private(set) var shape: [Int]
    public private(set) var strides: [Int]
    public let size: Int

    public init(storage: [Double], shape: [Int], strides: [Int]? = nil) {
        self.storage = storage
        self.shape = shape
        self.size = shape.reduce(1, *)

        self.strides = if let strides {
            strides
        } else {
            shape
                .reversed()
                .dropLast()
                .reduce(Array<Int>([1])) { [$0.first! * $1] + $0 }
        }
    }

    public subscript(_ s: [Int]) -> Double {
        get {
            storage[flatIndex(with: s)]
        }
        set {
            storage[flatIndex(with: s)] = newValue
        }
    }

    private func flatIndex(with indicies: [Int]) -> Int {
        precondition(indicies.count == shape.count, "Indicies and shape mismatch")
        return zip(indicies, strides).reduce(0, { $0 + $1.0 * $1.1 })
    }
}
