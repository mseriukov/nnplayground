public struct UniformDistribution<Element>: RandomDistribution where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    public let lowerBound: Element
    public let upperBound: Element

    public init(lowerBound: Element = 0, upperBound: Element = 1) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    public func next(using generator: inout RandomNumberGenerator) -> Element {
        Element.random(in: lowerBound..<upperBound, using: &generator)
    }

    func asAnyRandomDistribution() -> AnyRandomDistribution<Element> {
        AnyRandomDistribution(self)
    }
}
