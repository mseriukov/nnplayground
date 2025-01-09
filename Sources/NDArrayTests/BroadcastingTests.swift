import Testing
@testable import NDArray

@Suite("Broadcasting tests")
struct BroadcastingTests {
    @Test("Broadcast scalar")
    func broadcastScalar() throws {
        let storage = NDArrayStorage<Double>([42])
        let a = NDArray(storage: storage, shape: [])
        let b = a.broadcastTo([3, 3])!.makeContiguous()
        #expect(b.storage.data == [
            42, 42, 42,
            42, 42, 42,
            42, 42, 42
        ])
    }

    @Test("Broadcast rows")
    func broadcastRows() throws {
        let storage = NDArrayStorage<Double>([1, 2, 3])
        let a = NDArray(storage: storage, shape: [1, 3])
        let b = a.broadcastTo([3, 3])!.makeContiguous()
        #expect(b.storage.data == [
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        ])
    }

    @Test("Broadcast cols")
    func broadcastCols() throws {
        let storage = NDArrayStorage<Double>([
            1,
            2,
            3
        ])
        let a = NDArray(storage: storage, shape: [3, 1])
        let b = a.broadcastTo([3, 3])!.makeContiguous()
        #expect(b.storage.data == [
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ])
    }
}
