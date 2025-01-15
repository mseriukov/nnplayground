import Foundation

public enum Distribution<Element: BinaryFloatingPoint> {
    case uniform(lowerBound: Element = 0, upperBound: Element = 1)
    case normal(mean: Element = 0, standardDeviation: Element = 1)
    case kaiming(channels: Int)
}

extension Tensor where Element.RawSignificand: FixedWidthInteger {
    public static func random(
        shape: [Int],
        distribution: Distribution<Element>,
        generator: inout RandomNumberGenerator
    ) -> Tensor {
        let concreteDistribution = createDistribution(from: distribution)
        let totalCount = shape.reduce(1, *)
        let data: [Element] = (0..<totalCount).map { _ in
            Element(concreteDistribution.next(using: &generator))
        }
        return Tensor(
            storage: TensorStorage(data),
            shape: shape
        )
    }

    private static func createDistribution(
        from distribution: Distribution<Element>
    ) -> any RandomDistribution<Element> {
        switch distribution {
        case let .uniform(lowerBound, upperBound):
            UniformDistribution(
                lowerBound: lowerBound,
                upperBound: upperBound
            )

        case let .normal(mean, standardDeviation):
            NormalDistribution(
                mean: mean,
                standardDeviation: standardDeviation
            )

        case let .kaiming(channels):
            KaimingDistribution(
                channels: channels
            )
        }
    }
}
