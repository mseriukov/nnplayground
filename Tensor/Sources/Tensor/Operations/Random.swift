import Foundation

public enum Distribution {
    case uniform(lowerBound: Tensor.Element = 0, upperBound: Tensor.Element = 1)
    case normal(mean: Tensor.Element = 0, standardDeviation: Tensor.Element = 1)
    case kaiming(channels: Int)
}

extension Tensor {
    public static func random(
        shape: [Int],
        distribution: Distribution,
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
        from distribution: Distribution
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
