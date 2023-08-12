import XCTest
@testable import AlgebraKit

final class MatrixTests: XCTestCase {
    func test_matmul() throws {
        let m1 = Matrix(rows: 3, cols: 4, data: [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 8.0, 7.0, 6.0
        ])
        let m2 = Matrix(rows: 4, cols: 3, data: [
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0
        ])

        let m3 = m1 * m2

        XCTAssertEqual(m3.storage, [
            30.0, 30.0, 30.0,
            70.0, 70.0, 70.0,
            70.0, 70.0, 70.0
        ])
    }

    func test_matmul2() throws {
        let m1 = Matrix(rows: 1, cols: 3, data: [
            1.0, 2.0, 3.0
        ])
        let m2 = Matrix(rows: 1, cols: 3, data: [
            4.0, 5.0, 6.0
        ])

        let m3 = Matrix.matmul(m1: m1.transposed(), m2: m2)

        XCTAssertEqual(m3.storage, [
            4.0,  5.0,  6.0,
            8.0,  10.0, 12.0,
            12.0, 15.0, 18.0
        ])
    }
}



