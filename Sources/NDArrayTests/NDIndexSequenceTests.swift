import Testing
@testable import NDArray

@Suite("NDIndex tests")
struct NDIndexTests {
    @Test("All indicies are covered")
    func allIndiciesAreCovered() throws {
        let indexSeq = NDIndexSequence(shape: [3, 3, 3])
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

    @Test("Nil if shape is empty")
    func nilIfShapeIsEmpty() throws {
        var indexSeq = NDIndexSequence(shape: [])
        #expect(indexSeq.next() == nil)
    }
}