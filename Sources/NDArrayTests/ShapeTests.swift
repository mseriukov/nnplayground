import Testing
@testable import NDArray

@Suite("Shape tests")
struct ShapeTests {
    @Test("Shape")
    func shape() throws {
        let storage = NDArrayStorage(size: 1, initialValue: 0)
        #expect(NDArray(storage: storage, shape: [2, 5]).strides == [5, 1])
        #expect(NDArray(storage: storage, shape: [1, 2, 3]).strides == [6, 3, 1])
        #expect(NDArray(storage: storage, shape: [5, 3, 1, 4, 9]).strides == [108, 36 ,36, 9, 1])
        #expect(NDArray(storage: storage, shape: [1, 1]).strides == [1, 1])
    }
}
