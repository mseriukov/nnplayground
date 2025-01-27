import Testing
@testable import Tensor

@Suite
struct TransposedTests {
    @Test
    func testBasicTranspose() throws {
        let a = Tensor(
            storage: [
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ],
            shape: [3, 3]
        )
        let b = a.transposed().makeContiguous()
        #expect(b.storage.data == [
            1, 4, 7,
            2, 5, 8,
            3, 6, 9
        ])
    }

    @Test
    func testBasicTranspose2() throws {
        let a = Tensor(
            storage: [
                1, 2, 3, 1,
                4, 5, 6, 8,
                7, 8, 9, 0,
            ],
            shape: [3, 4]
        )
        let b = a.transposed().makeContiguous()
        #expect(b.shape == [4, 3])
        #expect(b.storage.data == [
            1, 4, 7,
            2, 5, 8,
            3, 6, 9,
            1, 8, 0,
        ])
    }

    @Test
    func testSlicedTranspose() throws {
        let a = Tensor(
            storage: [
                1, 2, 3,
            // ----------
                4, 5, 6,
                7, 8, 9
            ],
            shape: [3, 3]
        )
        let b = a.slice(start: [1, 0], shape: [2, 3]).transposed()
        #expect(b.shape == [3, 2])
        #expect(b.flatDataSlice == [
            4, 7,
            5, 8,
            6, 9
        ])
    }
}
