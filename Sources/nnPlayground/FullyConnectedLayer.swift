import AlgebraKit
import Accelerate

class FullyConnectedLayer: Layer {
    let inputSize: Int
    let outputSize: Int
    let activation: Activation

    // [rows       x columns  ]
    // [outputSize x inputSize]
    private(set) var weight: Matrix
    private(set) var wgrad: Matrix?

    // [rows x columns   ]
    // [1    x outputSize]
    private(set) var bias: Matrix
    private(set) var bgrad: Matrix?

    // [rows x columns  ]
    // [1    x inputSize]
    private(set) var input: Matrix

    // [rows x columns   ]
    // [1    x outputSize]
    private(set) var output: Matrix

    // [rows x columns  ]
    // [1    x inputSize]
    private(set) var weighedInput: Matrix

    init(
        inputSize: Int,
        outputSize: Int,
        activation: Activation
    ) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation

        weight = Matrix.random(rows: outputSize, cols: inputSize)
        bias = Matrix(rows: 1, cols: outputSize)

        weighedInput = Matrix(rows: 1, cols: outputSize)

        input = Matrix(rows: 1, cols: inputSize)
        output = Matrix(rows: 1, cols: outputSize)

        resetGrad()
    }

    func forward(input: Matrix) {
        assert(input.cols == inputSize, "Input size \(input.cols) doesn't match expected \(inputSize).")
        self.input = input
        weighedInput = Matrix.matmul(m1: weight, m2: input.transposed()) + bias
        output = activation.forward(weighedInput).transposed()
    }

    func backward(localGradient: Matrix) {
        assert(localGradient.cols == outputSize, "Loss local gradint size \(localGradient.cols) doesn't match expected \(outputSize).")
        let dL = localGradient.transposed() * activation.backward(weighedInput)
        wgrad = Matrix.matmul(m1: dL, m2: input)
        bgrad = dL
    }

    func resetGrad() {
        wgrad = Matrix(as: weight)
        bgrad = Matrix(as: bias)
    }
}
