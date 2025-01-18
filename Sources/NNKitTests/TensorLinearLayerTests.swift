import Testing
import AlgebraKit
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
        #expect(layer.weights.value.shape == [2, 3])
        #expect(layer.bias?.value.shape == [2])

        layer.weights.value = Tensor([2, 3], [
            1, 2, 3,
            4, 5, 6
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
