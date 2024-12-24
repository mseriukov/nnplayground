import AlgebraKit

public class Parameter {
    public var value: Matrix
    public var grad: Matrix

    init(size: Size) {
        value = Matrix(size: size, repeating: 0)
        grad = Matrix(as: value, repeating: 0)
    }

    public func randomize(_ kind: RandomKind, seed: UInt32? = nil) {
        value = Matrix.random(as: value, kind: kind, seed: seed)
    }

    public func resetGrad() {
        grad = Matrix(as: value, repeating: 0)
    }
}
