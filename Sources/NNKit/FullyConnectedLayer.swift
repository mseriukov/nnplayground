import AlgebraKit
import Accelerate

public class FullyConnectedLayer: Layer {
    public let inputSize: Int
    public let outputSize: Int
    public let activation: Activation

    // [rows       x columns   ]
    // [inputSize  x outputSize]
    private(set) public var weight: Matrix
    private(set) public var wgrad: Matrix = Matrix(rows: 1, cols: 1)

    // [rows x columns   ]
    // [1    x outputSize]
    private(set) public var bias: Matrix
    private(set) public var bgrad: Matrix = Matrix(rows: 1, cols: 1)

    // [rows x columns  ]
    // [1    x inputSize]
    private(set) public var input: Matrix

    // [rows x columns   ]
    // [1    x outputSize]
    private(set) public var output: Matrix

    // [rows x columns  ]
    // [1    x outputSize]
    private(set) public var weighedInput: Matrix

    public init(
        inputSize: Int,
        outputSize: Int,
        activation: Activation,
        weight: Matrix? = nil,
        bias: Matrix? = nil
    ) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation

        self.weight = weight ?? Matrix.random(rows: inputSize, cols: outputSize)
        self.bias = bias ?? Matrix.random(rows: 1, cols: outputSize)

        weighedInput = Matrix(rows: 1, cols: outputSize)

        input = Matrix(rows: 1, cols: inputSize)
        output = Matrix(rows: 1, cols: outputSize)

        resetGrad()
    }

    public func forward(_ input: Matrix) -> Matrix {
        assert(input.cols == inputSize, "Input size \(input.cols) doesn't match expected \(inputSize).")
        self.input = input
        weighedInput = Matrix.matmul(m1: weight.transposed(), m2: input.transposed()).transposed() + bias
        return activation.forward(weighedInput)
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        assert(localGradient.cols == outputSize, "Loss local gradint size \(localGradient.cols) doesn't match expected \(outputSize).")
        let dL = Matrix.elementwiseMul(m1: localGradient, m2: activation.backward(weighedInput))
        wgrad = Matrix.matmul(m1: dL.transposed(), m2: input).transposed()
        bgrad = dL
        return (weight * dL.transposed()).transposed()
    }

    public func updateParameters(learningRate: Float) {
        weight = weight - wgrad * learningRate
        bias = bias - bgrad * learningRate
        resetGrad()
    }

    private func resetGrad() {
        wgrad = Matrix(as: weight)
        bgrad = Matrix(as: bias)
    }
}
