//
//  PerformOperationSlow.swift
//  nnplayground
//
//  Created by Mikhail Seriukov on 18/01/2025.
//

extension Tensor {
    static func performOperationSlow(_ lhs: Self, _ rhs: Self, _ operation: (Element, Element) -> Element) -> Self {
        guard let resultShape = Tensor.broadcastShapes(lhs.shape, rhs.shape) else {
            fatalError("Shapes doesn't match and can't be broadcasted")
        }

        let lhs = lhs.broadcastTo(resultShape)!
        let rhs = rhs.broadcastTo(resultShape)!
        var result = Tensor.init(zeros: resultShape)
        result.forEachIndex {
            result.assign(operation(lhs[$0], rhs[$0]), at: $0)
        }
        return result
    }
}
