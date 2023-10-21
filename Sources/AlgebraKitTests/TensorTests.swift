import XCTest
@testable import AlgebraKit

final class TensorTests: XCTestCase {
    func test_strides() throws {
        let t = Tensor(Array(repeating: 0, count: 5 * 3 * 4 * 9), shape: [5, 3, 1, 4, 9])

        XCTAssertEqual(t.strides, [108, 36 ,36, 9, 1])
    }
}



