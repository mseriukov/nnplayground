import Foundation

public struct NormalDistribution<Element>: RandomDistribution where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    public let mean: Element
    public let standardDeviation: Element

    public init(mean: Element = 0, standardDeviation: Element = 1) {
        self.mean = mean
        self.standardDeviation = standardDeviation
    }

    public func next(using generator: inout RandomNumberGenerator) -> Element {
        let u1 = Element.random(in: 0..<1, using: &generator)
        let u2 = Element.random(in: 0..<1, using: &generator)
        let z0 = Element(
            sqrt(-2.0 * log(Double(u1))) * cos(2.0 * .pi * Double(u2))
        )
        return mean + standardDeviation * z0
    }

    func asAnyRandomDistribution() -> AnyRandomDistribution<Element> {
        AnyRandomDistribution(self)
    }
}
