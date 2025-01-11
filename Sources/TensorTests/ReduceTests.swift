import Testing
@testable import Tensor

@Suite
struct ReduceTests {
    @Test
    func testSumHorizontally() throws {
        let a = Tensor(
            storage: TensorStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )
        let b = a.sum(alongAxis: 1, keepDims: true)

        #expect(b.storage.data == [
             6,
            15,
            24,
        ])
        #expect(b.shape == [3, 1])
    }

    @Test
    func testSumVertically() throws {
        let a = Tensor(
            storage: TensorStorage([
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ]),
            shape: [3, 3]
        )
        let b = a.sum(alongAxis: 0, keepDims: true)

        #expect(b.storage.data == [
            12, 15, 18,
        ])
        #expect(b.shape == [1, 3])
    }
}
