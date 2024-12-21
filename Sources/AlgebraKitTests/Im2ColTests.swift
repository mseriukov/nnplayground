import XCTest
@testable import AlgebraKit

final class Im2ColTests: XCTestCase {
    func test_matmul() throws {
        let m1 = Matrix(rows: 4, cols: 4, data: [
             0.0,  1.0,  2.0,  3.0,
             4.0,  5.0,  6.0,  7.0,
             8.0,  9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 15.0,
        ])

        XCTAssertEqual(im2col(m1, 3), Matrix(rows: 9, cols: 4, data: [
             0.0,  1.0,  4.0,  5.0,
             1.0,  2.0,  5.0,  6.0,
             2.0,  3.0,  6.0,  7.0,
             4.0,  5.0,  8.0,  9.0,
             5.0,  6.0,  9.0, 10.0,
             6.0,  7.0, 10.0, 11.0,
             8.0,  9.0, 12.0, 13.0,
             9.0, 10.0, 13.0, 14.0,
            10.0, 11.0, 14.0, 15.0
        ]))
    }
}



