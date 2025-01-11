import Testing
@testable import Tensor

@Suite
struct TensorIndexSequenceTests {
    @Test
    func allIndiciesAreCovered() throws {
        let indexSeq = TensorIndexSequence(shape: [3, 3, 3])
        let expected = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
        ]

        for (i, indicies) in indexSeq.enumerated() {            
            #expect(indicies == expected[i])
        }
    }

    @Test
    func nilIfShapeIsEmpty() throws {
        var indexSeq = TensorIndexSequence(shape: [])
        #expect(indexSeq.next() == nil)
    }
}
