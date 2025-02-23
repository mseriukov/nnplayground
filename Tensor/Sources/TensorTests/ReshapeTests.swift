import Testing
@testable import Tensor

@Suite
struct ReshapeTests {
    @Test
    func simpleReshape() throws {
        let a = Tensor(
            storage: TensorStorage([
                0, 1, 2,
                3, 4, 5
            ]),
            shape: [2, 3]
        )

        let res = try a.reshape([3, 2])

        #expect(res.shape == [3, 2])
        #expect(res.strides == [2, 1])
        #expect(res.storage === a.storage, "Storage should not be copied.")
    }

    @Test
    func singletonDimensionReshape() throws {
        let a = Tensor(
            storage: TensorStorage([
                0, 1, 2,
                3, 4, 5
            ]),
            shape: [2, 3]
        )

        let res = try a.reshape([1, 2, 3])

        #expect(res.shape == [1, 2, 3])
        #expect(res.strides == [6, 3, 1])
    }

    @Test
    func nonContiguousReshape() throws {
        let a = Tensor(
            storage: TensorStorage([
                0, 1, 2,
                3, 4, 5
            ]),
            shape: [2, 3]
        )

        let nonContiguous = a.slice(start: [0, 0], shape: [2, 2])

        let res = try nonContiguous.reshape([4])

        #expect(res.isContiguous)
        #expect(res.shape == [4])
        #expect(res.strides == [1])
    }

    @Test
    func scalarReshape() throws {
        let a = Tensor([], [42])
        let res = try a.reshape([1])

        #expect(res.shape == [1])
        #expect(res.strides == [1])
    }

    @Test
    func invalidReshape() throws {
        let a = Tensor(
            storage: TensorStorage([
                0, 1, 2,
                3, 4, 5
            ]),
            shape: [2, 3]
        )
        #expect(throws: Tensor.ReshapeError.sizeMismatch, performing: {
            _ = try a.reshape([4, 4])
        })
    }
}
