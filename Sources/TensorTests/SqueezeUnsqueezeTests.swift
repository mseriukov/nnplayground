import Testing
@testable import Tensor

@Suite
struct SqueezeUnsqueezeTests {
    @Test
    func testSimpleSqueeze() throws {
        var a = Tensor(
            storage: [
                1, 2,
                3, 4,
            ],
            shape: [2, 1, 2, 1]
        )

        a.squeeze()

        #expect(a.shape == [2, 2])
        #expect(a.strides == [2, 1])
    }

    @Test
    func testSimpleSqueezeOneAxis() throws {
        var a = Tensor(
            storage: [
                1, 2,
                3, 4,
            ],
            shape: [2, 1, 2, 1]
        )

        a.squeeze([3])

        #expect(a.shape == [2, 1, 2])
        #expect(a.strides == [2, 2, 1])
    }

//    @Test
//    func testSimpleSqueeze() throws {
//        var a = Tensor(
//            storage: [
//                1, 2, 3,
//                4, 5, 6,
//                7, 8, 9,
//            ],
//            shape: [3, 1, 3, 1]
//        )
//
//        a.squeeze(axes: [])
//
//        #expect(a.shape == [2, 2])
//        #expect(a.strides == [2, 1])
//    }
}
