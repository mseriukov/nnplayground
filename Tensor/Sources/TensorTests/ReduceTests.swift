import Testing
@testable import Tensor

@Suite
struct ReduceTests {
    let _3x3 = Tensor(
        storage: TensorStorage([
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]),
        shape: [3, 3]
    )

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
    func test3x3SumHorizontally() throws {
        let res = _3x3.sum(alongAxis: 1)
        #expect(Array(res.dataSlice) == [
             6,
            15,
            24,
        ])
        #expect(res.shape == [3])
    }

    @Test
    func test3x3SumHorizontallyKeepDims() throws {
        let res = _3x3.sum(alongAxis: 1, keepDims: true)
        #expect(Array(res.dataSlice) == [
             6,
            15,
            24,
        ])
        #expect(res.shape == [3, 1])
    }

    @Test
    func test3x3SumVertically() throws {
        let res = _3x3.sum(alongAxis: 0)
        #expect(Array(res.dataSlice) == [
            12, 15, 18,
        ])
        #expect(res.shape == [3])
    }

    @Test
    func test3x3SumVerticallyKeepDims() throws {
        let res = _3x3.sum(alongAxis: 0, keepDims: true)
        #expect(Array(res.dataSlice) == [
            12, 15, 18,
        ])
        #expect(res.shape == [1, 3])
    }

    @Test
    func test4x4SumZero() throws {
        let res = _4x4x4.sum(alongAxis: 0, keepDims: true)
        #expect(Array(res.dataSlice) == [
            100, 104, 108, 112,
            116, 120, 124, 128,
            132, 136, 140, 144,
            148, 152, 156, 160
        ])
        #expect(res.shape == [1, 4, 4])
    }

    @Test
    func test4x4SumOne() throws {
        let res = _4x4x4.sum(alongAxis: 1, keepDims: true)
        #expect(Array(res.dataSlice) == [
             28,  32,  36,  40,

             92,  96, 100, 104,

            156, 160, 164, 168,
             
            220, 224, 228, 232
        ])
        #expect(res.shape == [4, 1, 4])
    }

    @Test
    func test4x4SumTwo() throws {
        let res = _4x4x4.sum(alongAxis: 2, keepDims: true)
        #expect(Array(res.dataSlice) == [
            10,
            26,
            42,
            58,

            74,
            90,
            106,
            122,

            138,
            154,
            170,
            186,

            202,
            218,
            234,
            250
        ])
        #expect(res.shape == [4, 4, 1])
    }

    @Test
    func testSumAlongAxes() throws {
        let a = Tensor(shape: [2, 3, 4, 5], value: 1)
        let res = a.sum(alongAxes: [1, 2], keepDims: false)

        #expect(res.shape == [2, 5])
        // [2, 3, 4, 5]
        // 1 1 1 1 1  1 1 1 1 1  1 1 1 1 1  1 1 1 1 1   1 1 1 1 1  1 1 1 1 1  1 1 1 1 1  1 1 1 1 1   1 1 1 1 1  1 1 1 1 1  1 1 1 1 1  1 1 1 1 1
        // 1 1 1 1 1  1 1 1 1 1  1 1 1 1 1  1 1 1 1 1   1 1 1 1 1  1 1 1 1 1  1 1 1 1 1  1 1 1 1 1   1 1 1 1 1  1 1 1 1 1  1 1 1 1 1  1 1 1 1 1
        // v
        // [2, 4, 5]
        // 3 3 3 3 3  3 3 3 3 3  3 3 3 3 3  3 3 3 3 3
        // 3 3 3 3 3  3 3 3 3 3  3 3 3 3 3  3 3 3 3 3
        // v
        // [2, 5]
        // 12 12 12 12 12
        // 12 12 12 12 12
        #expect(Array(res.dataSlice) == [
            12, 12, 12, 12, 12,
            12, 12, 12, 12, 12
        ])
    }
}
