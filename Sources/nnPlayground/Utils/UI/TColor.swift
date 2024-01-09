import Foundation

public struct TColor24 {
    public let r: UInt8
    public let g: UInt8
    public let b: UInt8

    public init(r: UInt8, g: UInt8, b: UInt8) {
        self.r = r
        self.g = g
        self.b = b
    }

    public static var red = TColor24(r: 255, g: 0, b: 0)
    public static var green = TColor24(r: 0, g: 255, b: 0)
}

public extension String {
    func fc(_ c: TColor24) -> String {
        "\u{001B}[38;2;\(c.r);\(c.g);\(c.b)m" + self
    }

    func bc(_ c: TColor24) -> String {
        "\u{001B}[48;2;\(c.r);\(c.g);\(c.b)m" + self
    }

    func reset() -> String {
        self + "\u{001B}[0m"
    }
}
