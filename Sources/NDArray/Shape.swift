public struct Shape {
    public private(set) var shape: [Int]
    public private(set) var strides: [Int]
    public let size: Int

    public init() {
        shape = []
        strides = []
        size = 1
    }

    public init(_ shape: [Int]) {
        self.shape = shape

        strides = shape
            .reversed()
            .dropLast()
            .reduce(Array<Int>([1])) { acc, dim in
                return acc + [acc.last! * dim]
            }.reversed()

        size = shape.reduce(1, *)
    }

    public func flatIndex(with indicies: [Int]) -> Int {        
        precondition(indicies.count == shape.count, "Indicies and shape mismatch")
        return zip(indicies, strides).reduce(0, { $0 + $1.0 * $1.1 })
    }
}

extension Shape: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Int...) {
        self.init(elements)
    }
}
