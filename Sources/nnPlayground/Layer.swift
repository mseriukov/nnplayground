import AlgebraKit

protocol Layer {
    var weight: Matrix { get }
    var output: Matrix { get }
    
    func forward(input: Matrix)
    func backward(localGradient: Matrix) -> Matrix

    func updateWeights(eta: Float)
    func resetGrad()
}
