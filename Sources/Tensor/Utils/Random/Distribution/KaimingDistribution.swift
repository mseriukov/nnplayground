import Foundation

public struct KaimingDistribution<Element>: RandomDistribution where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    public let channels: Int
    private let normalDistribution = NormalDistribution<Element>(mean: 0, standardDeviation: 1)

    public init(channels: Int) {
        self.channels = channels
    }

    public func next(using generator: inout RandomNumberGenerator) -> Element {
        sqrt(2.0 / Element(channels)) * normalDistribution.next(using: &generator)
    }
}
