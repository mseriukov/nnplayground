import ArgumentParser
import Foundation
import cnnutils
import AlgebraKit

@main
struct nnplayground: ParsableCommand {
    mutating func run() throws {


        let layer = FullyConnectedLayer(inputSize: 3, outputSize: 2, activation: .sigmoid)

        layer.forward(input: Matrix([1, 2, 3]))
        layer.backward(localGradient: Matrix([1, 2]))

        
    }
}
