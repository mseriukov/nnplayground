import AlgebraKit

public class LinearLayer: Layer {
    public let inputSize: Int
    public let outputSize: Int

    // [rows       x columns   ]
    // [inputSize  x outputSize]
    private(set) public lazy var weight: Parameter = {
        .init(size: Size(inputSize, outputSize))
    }()

    // [rows x columns   ]
    // [1    x outputSize]
    private(set) public lazy var bias: Parameter = {
        .init(size: Size(1, outputSize))
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

        weighedInput = Matrix(size: Size(1, outputSize))
        input = Matrix(size: Size(1, inputSize))
        output = Matrix(size: Size(1, outputSize))
    }

    public func forward(_ input: Matrix) -> Matrix {
        assert(input.size.cols == inputSize, "Input size \(input.size.cols) doesn't match expected \(inputSize).")
        self.input = input
        return matmul(weight.value.transposed(), input.transposed()).transposed() + bias.value
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        assert(localGradient.size.cols == outputSize, "Loss local gradint size \(localGradient.size.cols) doesn't match expected \(outputSize).")
        weight.grad += matmul(localGradient.transposed(), input).transposed()
        bias.grad += localGradient
        return (weight.value * localGradient.transposed()).transposed()
    }
}
