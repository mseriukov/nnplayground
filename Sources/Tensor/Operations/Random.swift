public enum Distribution<Element: BinaryFloatingPoint> {
    case uniform(lowerBound: Element = 0, upperBound: Element = 1)
    case normal(mean: Element = 0, standardDeviation: Element = 1)
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
            concreteDistribution.next(using: &generator) as! Element
        }
        return Tensor(
            storage: TensorStorage(data),
            shape: shape
        )
    }

    private static func createDistribution<E>(
        from distribution: Distribution<E>
    ) -> some RandomDistribution where E: BinaryFloatingPoint, E.RawSignificand: FixedWidthInteger {
        switch distribution {
        case let .uniform(lowerBound, upperBound):
            return UniformDistribution(lowerBound: lowerBound, upperBound: upperBound).asAnyRandomDistribution()
        case let .normal(mean, standardDeviation):
            return NormalDistribution(mean: mean, standardDeviation: standardDeviation).asAnyRandomDistribution()
        }
    }
}
