import AlgebraKit

protocol Layer {
    var output: Matrix { get }
    
    func forward(input: Matrix)
    func backward(localGradient: Matrix)
    func resetGrad()
}
