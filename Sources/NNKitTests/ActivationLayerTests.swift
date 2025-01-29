import Testing
@testable import Tensor
@testable import NNKit

@Suite
struct ActivationLayerTests {
    @Test
    func testSoftmaxForward() throws {
        let input = Tensor([2, 3], [1, 2, 3, 4, 5, 6])
        let softmaxLayer = TensorActivationLayer(.softmax)

        let output = softmaxLayer.forward(input)
        let rowSums = output.sum(alongAxis: output.shape.count - 1)

        #expect(rowSums.isApproximatelyEqual(to: Tensor(shape: rowSums.shape, value: 1)))

        // Verify value range (0 <= output <= 1)
        for value in Array(output.dataSlice) {
            #expect(value >= 0 && value <= 1)
        }
    }

    @Test
    func testSoftmaxForwardZeroes() throws {
        let input = Tensor([1, 3], [0, 0, 0])
        let softmaxLayer = TensorActivationLayer(.softmax)

        let output = softmaxLayer.forward(input)

        let expected: [Tensor.Element] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        #expect(output.isApproximatelyEqual(to: Tensor([1, 3], expected)))
    }

    @Test
    func testSoftmaxBackward() throws {
        let input = Tensor([1, 3], [1, 2, 3])
        let localGradient = Tensor([1, 3], [1, 1, 1])

        let softmaxLayer = TensorActivationLayer(.softmax)
        _ = softmaxLayer.forward(input)
        let analyticalGradient = softmaxLayer.backward(localGradient)
        let numericalGrad = input.numericalGradient(forwardPass: softmaxLayer.forward)

        #expect(analyticalGradient.isApproximatelyEqual(to: numericalGrad))
    }
}
