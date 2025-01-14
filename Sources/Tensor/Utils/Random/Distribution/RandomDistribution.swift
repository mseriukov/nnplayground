public protocol RandomDistribution {
    associatedtype Element: BinaryFloatingPoint where Element.RawSignificand: FixedWidthInteger
    func next(using generator: inout RandomNumberGenerator) -> Element
}

public struct AnyRandomDistribution<Element>: RandomDistribution where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    private let _next: (inout RandomNumberGenerator) -> Element

    init<T: RandomDistribution>(_ distribution: T) where T.Element == Element {
        self._next = distribution.next
    }

    public func next(using generator: inout RandomNumberGenerator) -> Element {
        _next(&generator)
    }
}
