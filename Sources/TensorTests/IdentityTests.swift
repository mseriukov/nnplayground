import Testing
@testable import Tensor

@Suite
struct IdentityTests {
    @Test
    func test3x3() throws {
        let a = Tensor.identity(rank: 2, size: 3)
        #expect(Array(a.dataSlice) == [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ])
    }

    @Test
    func test3x3x3() throws {
        let a = Tensor.identity(rank: 3, size: 3)
        #expect(Array(a.dataSlice) == [
            1, 0, 0,
            0, 0, 0,
            0, 0, 0,

            0, 0, 0,
            0, 1, 0,
            0, 0, 0,

            0, 0, 0,
            0, 0, 0,
            0, 0, 1
        ])
    }
}
