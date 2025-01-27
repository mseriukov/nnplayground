import Testing
@testable import Tensor

@Suite
struct OperatorsTests {
    @Test
    func subtractScalarRight() throws {
        let a = Tensor(
            storage: TensorStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )

        let res = a - 1

        #expect(res.storage.data == [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
        ])
    }

    @Test
    func subtractFromScalar() throws {
        let a = Tensor(
            storage: TensorStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )

        let res = 1 - a

        #expect(res.storage.data == [
            -0, -1, -2,
            -3, -4, -5,
            -6, -7, -8
        ])
    }

    @Test
    func testIncrementedBy() throws {
        let a = Tensor(
            storage: TensorStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )


        let b = a.incremented(by: 42)

        #expect(b.storage.data == [
            43, 44, 45,
            46, 47, 48,
            49, 50, 51
        ])
    }

    @Test
    func testSliceIncrementedBy() throws {
        let a = Tensor(
            storage: TensorStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )


        let b = a.slice(start: [1, 0], shape: [1, 3]).incremented(by: 42)

        #expect(b.flatDataSlice == [
            46, 47, 48,
        ])
    }
}
