public struct Stride: Hashable {
    public var horizontal: Int
    public var vertical: Int

    public init(horizontal: Int, vertical: Int) {
        self.horizontal = horizontal
        self.vertical = vertical
    }
}

extension Stride: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.horizontal = value
        self.vertical = value
    }
}
