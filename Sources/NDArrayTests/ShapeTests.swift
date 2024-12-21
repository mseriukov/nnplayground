import XCTest
@testable import NDArray

final class ShapeTests: XCTestCase {
    func test_shape() throws {
        XCTAssertEqual(Shape([2, 5]).strides, [5, 1])
        XCTAssertEqual(Shape([1, 2, 3]).strides, [6, 3, 1])
        XCTAssertEqual(Shape([5, 3, 1, 4, 9]).strides, [108, 36 ,36, 9, 1])
        XCTAssertEqual(Shape([1, 1]).strides, [1, 1])
    }

    func test_index() throws {
        XCTAssertEqual(Shape([2, 5]).flatIndex(with: [1, 2]), 7)
        XCTAssertEqual(Shape([1, 2, 3]).flatIndex(with: [1, 2, 1]), 13)
        XCTAssertEqual(Shape([5, 3, 1, 4, 9]).flatIndex(with: [1, 2, 0, 0, 0]), 180)
        XCTAssertEqual(Shape([1, 1]).flatIndex(with: [0, 0]), 0)
        XCTAssertEqual(Shape().flatIndex(with: []), 0)
    }

    func test_broadcast() throws {
        let a = NDArray(
            storage: ArrayStorage([1,2,3,4]),
            shape: [2,2]
        )
        let b = ReshapedView(parent: a, shape: [4])

        XCTAssertEqual(b[3], a[1,1])
    }
}
