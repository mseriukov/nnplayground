import Tensor

public class Conv2DLayer: Layer {
    public var weights: Parameter
    public var bias: Parameter?
    public let kernelSize: (rows: Int, cols: Int)
    public let stride: (rows: Int, cols: Int)
    public let padding: (rows: Int, cols: Int)
    public let dilation: (rows: Int, cols: Int)

    private var cachedColInput: Tensor?
    private var cachedInputShape: [Int]?
    private var randomGenerator: any RandomNumberGenerator = SystemRandomNumberGenerator()

    private let outputChannels: Int

    public var parameters: [Parameter] {
        [weights, bias].compactMap { $0 }
    }

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: (Int, Int),
        stride: (Int, Int) = (1, 1),
        padding: (Int, Int) = (0, 0),
        dilation: (Int, Int) = (1, 1)
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.outputChannels = outputChannels

        let weightShape = [outputChannels, inputChannels, kernelSize.0, kernelSize.1]
        self.weights = Parameter(tensor: Tensor.random(
            shape: weightShape,
            distribution: .kaiming(channels: inputChannels),
            generator: &randomGenerator
        ))
        self.bias = Parameter(tensor: Tensor(zeros: [outputChannels]))
    }

    public func forward(_ input: Tensor) throws -> Tensor {
        // [batchSize, inputChannels, inputHeight, inputWidth]
        let (batchSize, inputRows, inputCols) = (input.shape[0], input.shape[2], input.shape[3])

        let outputRows = (inputRows + 2 * padding.rows - dilation.rows * (kernelSize.rows - 1) - 1) / stride.rows + 1
        let outputCols = (inputCols + 2 * padding.cols - dilation.cols * (kernelSize.cols - 1) - 1) / stride.cols + 1

        // [inputChannels * kernelRows * kernelCols, batchSize * outputRows * outputCols]
        let colInput = try input.im2col(kernelSize: kernelSize, stride: stride, padding: padding, dilation: dilation)
        cachedColInput = colInput
        cachedInputShape = input.shape

        // [outputChannels, inputChannels * kernelRows * kernelCols]
        let wshape = weights.value.shape
        let reshapedWeights = try weights.value.reshape([wshape[0], wshape[1] * wshape[2] * wshape[3]])
        let bias = bias?.value.unsqueezed(axis: 1) ?? Tensor(zeros: colInput.shape)

        // [outputChannels, batchSize * outputRows * outputCols]
        let output = reshapedWeights.matmul(colInput) + bias

        // [batchSize, outputChannels, outputRows, outputCols]
        return try output.reshape([batchSize, outputChannels, outputRows, outputCols])
    }

    public func backward(_ localGradient: Tensor) throws -> Tensor {
        // [batchSize, outputChannels, outputRows, outputCols]
        guard
            let colInput = cachedColInput,
            let inputShape = cachedInputShape
        else {
            fatalError("No cached input. Did you forget to perform a forward pass?")
        }
        // reshape weights to [outputChannels, inputChannels * kernelRows * kernelCols]
        let wshape = weights.value.shape
        let reshapedWeights = try weights.value.reshape([wshape[0], wshape[1] * wshape[2] * wshape[3]])

        // Compute weight gradient
        let weightGradient = localGradient.matmul(colInput.transposed())
        weights.gradient?.add(try weightGradient.reshape(weights.value.shape))

        // Compute bias gradient
        if var biasGrad = bias?.gradient {
            biasGrad.add(localGradient.sum(alongAxes: [0, 2, 3]))
        }

        // Compute input gradient using col2im
        let inputGradient = reshapedWeights.transposed().matmul(localGradient)
        return inputGradient.col2im(
            inputShape: inputShape,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation
        )
    }
}
