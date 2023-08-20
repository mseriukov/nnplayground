import AlgebraKit
import Accelerate
import Foundation

public class LinearLayer: Layer {
    public let inputSize: Int
    public let outputSize: Int

    // [rows       x columns   ]
    // [inputSize  x outputSize]
    private(set) public lazy var weight: Parameter = {
        .init(rows: inputSize, cols: outputSize)
    }()

    // [rows x columns   ]
    // [1    x outputSize]
    private(set) public lazy var bias: Parameter = {
        .init(rows: 1, cols: outputSize)
    }()

    // [rows x columns  ]
    // [1    x inputSize]
    private(set) public var input: Matrix

    // [rows x columns   ]
    // [1    x outputSize]
    private(set) public var output: Matrix

    // [rows x columns  ]
    // [1    x outputSize]
    private(set) public var weighedInput: Matrix

    public var parameters: [Parameter] {
        [weight, bias]
    }

    public init(
        inputSize: Int,
        outputSize: Int
    ) {
        self.inputSize = inputSize
        self.outputSize = outputSize

        weighedInput = Matrix(rows: 1, cols: outputSize)
        input = Matrix(rows: 1, cols: inputSize)
        output = Matrix(rows: 1, cols: outputSize)

        resetGrad()
    }

    public func forward(_ input: Matrix) -> Matrix {
        assert(input.cols == inputSize, "Input size \(input.cols) doesn't match expected \(inputSize).")
        self.input = input
        return matmul(weight.value.transposed(), input.transposed()).transposed() + bias.value
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        assert(localGradient.cols == outputSize, "Loss local gradint size \(localGradient.cols) doesn't match expected \(outputSize).")
        weight.grad = matmul(localGradient.transposed(), input).transposed()
        bias.grad = localGradient
        return (weight.value * localGradient.transposed()).transposed()
    }

    public func updateParameters(eta: Float) {
        // TODO: Move it away from layer to an optimizer.
        weight.value = weight.value - weight.grad * eta
        bias.value = bias.value - bias.grad * eta
        resetGrad()
    }

    private func resetGrad() {
        weight.grad = Matrix(as: weight.value)
        bias.grad = Matrix(as: bias.value)
    }
}
