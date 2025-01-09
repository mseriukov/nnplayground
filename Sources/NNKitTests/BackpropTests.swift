import Testing
import AlgebraKit
@testable import NNKit

struct BackpropTests {
    var nn: [Layer] = []

    @Test("Forward propagation")
    func test_forward() throws {
        var input = Matrix(size: Size(1, 2), data: [
            2.0, 3.0
        ])
        for l in nn {
            input = l.forward(input)
        }
        #expect(input.storage[0] == 2)
    }
}
