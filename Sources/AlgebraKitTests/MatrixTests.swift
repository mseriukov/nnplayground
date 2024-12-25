import XCTest
@testable import AlgebraKit

final class MatrixTests: XCTestCase {
    func test_matmul() throws {
        let m1 = Matrix(size: Size(3, 4), data: [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 8.0, 7.0, 6.0
        ])
        let m2 = Matrix(size: Size(4, 3), data: [
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
        let m1 = Matrix(size: Size(1, 3), data: [
            1.0, 2.0, 3.0
        ])
        let m2 = Matrix(size: Size(1, 3), data: [
            4.0, 5.0, 6.0
        ])

        let m3 = matmul(m1.transposed(), m2)

        XCTAssertEqual(m3.storage, [
            4.0,  5.0,  6.0,
            8.0,  10.0, 12.0,
            12.0, 15.0, 18.0
        ])
    }

    func test_padding_1() throws {
        var m1 = Matrix(size: 1, data: [
            1.0
        ])

        m1.pad(.init(top: 1, left: 1, bottom: 1, right: 1))

        XCTAssertEqual(m1.storage, [
            0.0,  0.0,  0.0,
            0.0,  1.0,  0.0,
            0.0,  0.0,  0.0
        ])
    }

    func test_padding_2() throws {
        var m1 = Matrix(size: 1, data: [
            1.0
        ])

        m1.pad(.init(top: 1, left: 2, bottom: 3, right: 4))

        XCTAssertEqual(m1.storage, [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
    }

    func test_join() throws {
        var m1 = Matrix(size: 3, data: [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ])

        var m2 = Matrix(size: 3, data: [
            2.0, 2.0, 2.0,
            2.0, 2.0, 2.0,
            2.0, 2.0, 2.0,
        ])

        var m3 = Matrix(size: 3, data: [
            3.0, 3.0, 3.0,
            3.0, 3.0, 3.0,
            3.0, 3.0, 3.0,
        ])

        XCTAssertEqual(
            mathjoin(m1, m2, m3),
            Matrix(size: Size(3, 9), data: [
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
            ])
        )
    }
}
