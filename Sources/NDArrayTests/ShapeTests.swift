import XCTest
@testable import NDArray

final class ShapeTests: XCTestCase {
    func test_shape() throws {
        XCTAssertEqual(NDArray(storage: [0.0], shape: [2, 5]).strides, [5, 1])
        XCTAssertEqual(NDArray(storage: [0.0], shape: [1, 2, 3]).strides, [6, 3, 1])
        XCTAssertEqual(NDArray(storage: [0.0], shape: [5, 3, 1, 4, 9]).strides, [108, 36 ,36, 9, 1])
        XCTAssertEqual(NDArray(storage: [0.0], shape: [1, 1]).strides, [1, 1])
    }
}
