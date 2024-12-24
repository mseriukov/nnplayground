public struct Size: Hashable {
    public var rows: Int
    public var cols: Int

    public init(_ size: Int) {
        self.init(size, size)
    }

    public init(_ rows: Int, _ cols: Int) {
        self.rows = rows
        self.cols = cols
    }

    public var elementCount: Int {
        rows * cols
    }
}

extension Size: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init(value)
    }
}
