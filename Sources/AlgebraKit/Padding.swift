public struct Padding {
    public var top: Int
    public var left: Int
    public var bottom: Int
    public var right: Int

    public init(top: Int, left: Int, bottom: Int, right: Int) {
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
    }
}

extension Padding: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.top = value
        self.left = value
        self.bottom = value
        self.right = value
    }
}
