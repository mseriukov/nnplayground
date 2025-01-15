import Tensor

extension Tensor {
    func numericalGradient(
        epsilon: Element = 1e-5,
        forwardPass: (Tensor<Element>) -> Element
    ) -> Tensor<Element> {
        var gradient = Tensor(zeros: shape)
        forEachIndex { index in
            var tensorPerturbed = self
            tensorPerturbed[index] += epsilon
            let lossPlus = forwardPass(tensorPerturbed)
            tensorPerturbed[index] -= 2 * epsilon
            let lossMinus = forwardPass(tensorPerturbed)
            gradient.assign((lossPlus - lossMinus) / (2 * epsilon), at: index)
        }
        return gradient
    }
}

