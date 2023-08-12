import Foundation
import AlgebraKit

enum Activation {
    case sigmoid

    var forward: (Matrix) -> Matrix {
        switch self {
        case .sigmoid:
            // This is probably slow as hell.
            
            return { Matrix(as: $0, data: ContiguousArray($0.storage.map { 1.0 / (1.0 + exp(-$0)) })) }
        }
    }

    var backward: (Matrix) -> Matrix {
        switch self {
        case .sigmoid:
            // This is probably slow as hell.
            return { forward($0) * (Matrix(as: $0, repeating: 1) - forward($0)) }
        }
    }
}
