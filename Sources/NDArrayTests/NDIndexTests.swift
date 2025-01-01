import XCTest
@testable import NDArray

final class NDIndexTests: XCTestCase {
    func test_index() throws {
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
            XCTAssertEqual(indicies, expected[i])
        }
    }
}
