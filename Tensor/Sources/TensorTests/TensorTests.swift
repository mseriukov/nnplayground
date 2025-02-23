import Testing
@testable import Tensor

@Suite
struct TensoryTests {
    @Test
    func mul() throws {
        var a = Tensor(
            storage: TensorStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )
        let b = Tensor(
            storage: TensorStorage([2]),
            shape: [1]
        )

        a.mulBroadcasted(b)

        #expect(Array(a.dataSlice) == [
             2,  4,  6,
             8, 10, 12,
            14, 16, 18
        ])
    }

    @Test
    func add() throws {
        var a = Tensor(
            storage: TensorStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )
        let b = Tensor(
            storage: TensorStorage([10]),
            shape: [1]
        )

        a.addBroadcasted(b)

        #expect(Array(a.dataSlice) == [
            11, 12, 13,
            14, 15, 16,
            17, 18, 19
        ])
    }
}
