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
}
