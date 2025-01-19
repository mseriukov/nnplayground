import Tensor

extension Tensor {
    func numericalGradient(
        epsilon: Tensor.Element = 1e-5,
        forwardPass: (Tensor) -> Tensor
    ) -> Tensor {
        var gradient = Tensor(zeros: shape)
        let loss: (Tensor) -> Tensor.Element = {
            $0.sum()[0]
        }
        forEachIndex { index in
            var tensorPerturbed = self
            tensorPerturbed[index] += epsilon
            let lossPlus = loss(forwardPass(tensorPerturbed))
            tensorPerturbed[index] -= 2 * epsilon
            let lossMinus = loss(forwardPass(tensorPerturbed))
            gradient.assign((lossPlus - lossMinus) / (2 * epsilon), at: index)
        }
        return gradient
    }
}

