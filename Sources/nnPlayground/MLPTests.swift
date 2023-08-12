import Foundation
import cnnutils
import AlgebraKit

class MLPTests {
    let network: [any Layer] = [
        FullyConnectedLayer(inputSize: 724, outputSize: 100, activation: .sigmoid),
        FullyConnectedLayer(inputSize: 100, outputSize: 10, activation: .sigmoid)
    ]

    func run() {
        let clock = ContinuousClock()
        let elapsed = clock.measure {
            for i in 0..<1000 {
                forward(input: Matrix.random(rows: 1, cols: 724))
            }
        }
        print(elapsed)
    }

    func forward(input: Matrix) {
        var input: Matrix = input
        for l in network {
            l.forward(input: input)
            input = l.output
        }
    }

    func backward(localGradient: Matrix) {

    }
}
