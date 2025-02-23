import Testing
import Tensor
@testable import Tensor
@testable import NNKit

@Suite
struct Conv2dLayerTests {
    @Test
    func testShape() throws {
        let batchSize = 2
        let inChannels = 3
        let outChannels = 2
        let inputSize = (5, 5)
        let kernelSize = (3, 3)

        let input = Tensor(shape: [batchSize, inChannels, inputSize.0, inputSize.1], value: 1)

        let convLayer = Conv2DLayer(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize)
        let output = try convLayer.forward(input)

        let expectedOutputShape = [batchSize, outChannels, inputSize.0 - kernelSize.0 + 1, inputSize.1 - kernelSize.1 + 1]
        #expect(output.shape == expectedOutputShape)

        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1

        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1

        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1
        // 1 1 1 1 1

    }
}
