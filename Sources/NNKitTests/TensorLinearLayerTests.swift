import Testing
@testable import Tensor
@testable import NNKit

@Suite
struct TensorLinearLayerTests {
    @Test()
    func testLinearLayer() throws {
        let layer = TensorLinearLayer(
            inputDim: 3,
            outputDim: 2,
            includeBias: true
        )
        #expect(layer.weights.value.shape == [3, 2])
        #expect(layer.bias?.value.shape == [2])

        layer.weights.value = Tensor([3, 2], [
            1, 4,
            2, 5,
            3, 6,
        ])
        layer.bias?.value = Tensor([2], [7, 9])

        let input = Tensor([2, 3], [
            1, 1, 1, // input 1
            2, 2, 2  // input 2
        ])

        let output = layer.forward(input)
        
        #expect(output.shape == [2, 2])
        #expect(output.storage.data == [
            13, 24, // output 1
            19, 39  // output 2
        ])
    }
}
