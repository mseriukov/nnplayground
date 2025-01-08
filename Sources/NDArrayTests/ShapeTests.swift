import XCTest
@testable import NDArray

final class ShapeTests: XCTestCase {
    func test_shape() throws {
        let storage = NDArrayStorage(size: 1, initialValue: 0)
        XCTAssertEqual(NDArray(storage: storage, shape: [2, 5]).strides, [5, 1])
        XCTAssertEqual(NDArray(storage: storage, shape: [1, 2, 3]).strides, [6, 3, 1])
        XCTAssertEqual(NDArray(storage: storage, shape: [5, 3, 1, 4, 9]).strides, [108, 36 ,36, 9, 1])
        XCTAssertEqual(NDArray(storage: storage, shape: [1, 1]).strides, [1, 1])
    }
}
