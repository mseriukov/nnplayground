import AlgebraKit

public protocol Layer {
    func forward(_ input: Matrix) -> Matrix
    func backward(_ localGradient: Matrix) -> Matrix
    func updateParameters(learningRate: Float)
}
