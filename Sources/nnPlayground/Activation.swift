import Foundation
import AlgebraKit

enum Activation {
    case sigmoid
    case relu
    case none

    var forward: (Matrix) -> Matrix {
        switch self {
        case .sigmoid:
            // This is probably slow as hell.
            return { Matrix(as: $0, data: ContiguousArray($0.storage.map { 1.0 / (1.0 + exp(-$0)) })) }

        case .relu:
            return { Matrix(as: $0, data: ContiguousArray($0.storage.map { max(0, $0) })) }

        case .none:
            return { $0 }
        }
    }

    var backward: (Matrix) -> Matrix {
        switch self {
        case .sigmoid:
            // This is probably slow as hell.
            return { Matrix.elementwiseMul(m1: forward($0), m2:  (Matrix(as: $0, repeating: 1) - forward($0))) }

        case .relu:
            return { Matrix(as: $0, data: ContiguousArray($0.storage.map { $0 > 0 ? 1.0 : 0.0 })) }

        case .none:
            return { $0 }
        }
    }
}
