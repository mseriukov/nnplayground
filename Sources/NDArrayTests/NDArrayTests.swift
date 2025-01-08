import XCTest
@testable import NDArray

final class NDArrayTests: XCTestCase {
    func test_mul() throws {
        var a = NDArray(
            storage: NDArrayStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )
        let b = NDArray(
            storage: NDArrayStorage([2]),
            shape: [1]
        )

        a.mulBroadcasted(b)

        XCTAssertEqual(a.storage.data, [
             2,  4,  6,
             8, 10, 12,
            14, 16, 18
        ])
    }

    func test_add() throws {
        var a = NDArray(
            storage: NDArrayStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )
        let b = NDArray(
            storage: NDArrayStorage([10]),
            shape: [1]
        )

        a.addBroadcasted(b)

        XCTAssertEqual(a.storage.data, [
            11, 12, 13,
            14, 15, 16,
            17, 18, 19
        ])
    }
}
