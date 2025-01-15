import Testing
@testable import Tensor
@testable import NNKit

@Suite
struct ActivationLayerTests {
    @Test
    func testSoftmaxForward() throws {
        let input = Tensor<Double>([2, 3], [1, 2, 3, 4, 5, 6])
        let softmaxLayer = TensorActivationLayer<Double>(.softmax)

        let output = softmaxLayer.forward(input)
        let rowSums = output.sum(alongAxis: output.shape.count - 1)

        #expect(rowSums.isApproximatelyEqual(to: Tensor(shape: rowSums.shape, value: 1)))

        // Verify value range (0 <= output <= 1)
        for value in output.storage.data {
            #expect(value >= 0 && value <= 1)
        }
    }

    @Test
    func testSoftmaxForwardZeroes() throws {
        let input = Tensor<Double>([1, 3], [0, 0, 0])
        let softmaxLayer = TensorActivationLayer<Double>(.softmax)

        let output = softmaxLayer.forward(input)

        #expect(output.isApproximatelyEqual(to: Tensor<Double>([1, 3], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])))
    }

    @Test
    func testSoftmaxBackward() throws {
        let input = Tensor<Double>([1, 3], [1, 2, 3])
        let localGradient = Tensor<Double>([1, 3], [0.1, 0.2, 0.3])

        let softmaxLayer = TensorActivationLayer<Double>(.softmax)
        _ = softmaxLayer.forward(input)
        let analyticalGradient = softmaxLayer.backward(localGradient)
        let numericalGrad = input.numericalGradient(forwardPass: {
            let softmaxOutput = softmaxLayer.forward($0)
            return (softmaxOutput * localGradient).sum()[0]
        })

        #expect(analyticalGradient.isApproximatelyEqual(to: numericalGrad))
    }
}
