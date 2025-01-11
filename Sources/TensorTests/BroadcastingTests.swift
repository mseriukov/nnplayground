import Testing
@testable import Tensor

@Suite
struct BroadcastingTests {
    @Test
    func broadcastScalar() throws {
        let a = Tensor(storage: [42], shape: [])
        let b = a.broadcastTo([3, 3])!.makeContiguous()
        #expect(b.storage.data == [
            42, 42, 42,
            42, 42, 42,
            42, 42, 42
        ])
    }

    @Test
    func broadcastRows() throws {
        let a = Tensor(storage: [1, 2, 3], shape: [1, 3])
        let b = a.broadcastTo([3, 3])!.makeContiguous()
        #expect(b.storage.data == [
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        ])
    }

    @Test
    func broadcastCols() throws {
        let a = Tensor(
            storage: [
                1,
                2,
                3
            ],
            shape: [3, 1]
        )
        let b = a.broadcastTo([3, 3])!.makeContiguous()
        #expect(b.storage.data == [
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ])
    }
}
