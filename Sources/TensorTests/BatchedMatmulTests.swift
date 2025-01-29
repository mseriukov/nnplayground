import Testing
@testable import Tensor

@Suite
struct BatchedMatmulTests {
    @Test
    func testBatchedMatmul() throws {
        let a = Tensor(
            storage: [
                 1,  2,
                 3,  4,

                 5,  6,
                 7,  8,

                 9, 10,
                11, 12
            ],
            shape: [3, 2, 2]
        )

        let b = Tensor(
            storage: [
                 1,  2,
                 3,  4,

                 5,  6,
                 7,  8,

                 9, 10,
                11, 12
            ],
            shape: [3, 2, 2]
        )

        let c = Tensor.batchedMatMul(a, b)

        #expect(Array(c.dataSlice) == [
              7,  10,
             15,  22,

             67,  78,
             91, 106,

            191, 210,
            231, 254
        ])
    }
}
