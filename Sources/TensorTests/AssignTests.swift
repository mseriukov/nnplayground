import Testing
@testable import Tensor

@Suite
struct AssignTests {
    let _4x4x4 = Tensor(
        storage: [
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,
            13, 14, 15, 16,

            17, 18, 19, 20,
            21, 22, 23, 24,
            25, 26, 27, 28,
            29, 30, 31, 32,

            33, 34, 35, 36,
            37, 38, 39, 40,
            41, 42, 43, 44,
            45, 46, 47, 48,

            49, 50, 51, 52,
            53, 54, 55, 56,
            57, 58, 59, 60,
            61, 62, 63, 64,
        ],
        shape: [4, 4, 4]
    )

    @Test
    func test2dMiddleAssign() throws {
        var a = Tensor(
            storage: [
                 1,  2,  3,  4,
                 5,  6,  7,  8,
                 9, 10, 11, 12,
                13, 14, 15, 16
            ],
            shape: [4, 4]
        )

        let b = Tensor(
            storage: [
                 0,  1,
                 2,  3,
            ],
            shape: [2, 2]
        )

        a.assign(start: [1, 1], tensor: b)
        #expect(Array(a.dataSlice) == [
            1,  2,  3,  4,
            5,  0,  1,  8,
            9,  2,  3, 12,
           13, 14, 15, 16
        ])
    }

    @Test
    func test3dMiddleAssign() throws {
        var a = _4x4x4


        let b = Tensor(
            storage: [
                 0,  1,
                 2,  3,

                 4,  5,
                 6,  7,
            ],
            shape: [2, 2, 2]
        )

        a.assign(start: [1, 1, 1], tensor: b)
        #expect(Array(a.dataSlice) == [
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
           13, 14, 15, 16,

           17, 18, 19, 20,
           21,  0,  1, 24,
           25,  2,  3, 28,
           29, 30, 31, 32,

           33, 34, 35, 36,
           37,  4,  5, 40,
           41,  6,  7, 44,
           45, 46, 47, 48,

           49, 50, 51, 52,
           53, 54, 55, 56,
           57, 58, 59, 60,
           61, 62, 63, 64,
       ])
    }
}
