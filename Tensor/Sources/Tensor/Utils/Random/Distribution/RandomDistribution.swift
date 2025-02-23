public protocol RandomDistribution<Element> {
    associatedtype Element: BinaryFloatingPoint where Element.RawSignificand: FixedWidthInteger
    func next(using generator: inout RandomNumberGenerator) -> Element
}
