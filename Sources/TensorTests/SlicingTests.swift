import Testing
@testable import Tensor

@Suite
struct SlicingTests {
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
    func test2dMiddleSlice() throws {
        let a = Tensor(
            storage: [
                 1,  2,  3,  4,
                 5,  6,  7,  8,
                 9, 10, 11, 12,
                13, 14, 15, 16
            ],
            shape: [4, 4]
        )
        let b = a.slice(
            start: [1, 1],
            shape: [2, 2]
        ).makeContiguous()
        #expect(b.storage.data == [
             6,  7,
            10, 11
        ])
    }

    @Test
    func test3dMiddleSlice() throws {
        let result = _4x4x4.slice(
            start: [1, 1, 1],
            shape: [2, 2, 2]
        ).makeContiguous()
        #expect(result.storage.data == [
            22, 23,
            26, 27,

            38, 39,
            42, 43,
        ])
    }

    @Test
    func test3dFirstPlaneSlice() throws {
        let result = _4x4x4.slice(
            start: [0, 0, 0],
            shape: [1, 4, 4]
        ).makeContiguous()
        print(result.shape)
        #expect(result.storage.data == [
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,
            13, 14, 15, 16,
        ])
    }

    @Test
    func test3dLastPlaneSlice() throws {
        let result = _4x4x4.slice(
            start: [3, 0, 0],
            shape: [1, 4, 4]
        ).makeContiguous()
        print(result.shape)
        #expect(result.storage.data == [
            49, 50, 51, 52,
            53, 54, 55, 56,
            57, 58, 59, 60,
            61, 62, 63, 64,
        ])
    }
}
