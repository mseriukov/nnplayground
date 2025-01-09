public struct Dilation: Hashable, Sendable {
    public var horizontal: Int
    public var vertical: Int

    public static let none = Self(horizontal: 1, vertical: 1)

    public init(horizontal: Int, vertical: Int) {
        self.horizontal = horizontal
        self.vertical = vertical
    }
}

extension Dilation: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.horizontal = value
        self.vertical = value
    }
}
