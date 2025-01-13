import Testing
@testable import Tensor

@Suite
struct StridesTests {
    @Test
    func testDefaultStrides() throws {
        #expect(Tensor<Double>.defaultStrides(for: [2, 5]) == [5, 1])
        #expect(Tensor<Double>.defaultStrides(for: [1, 2, 3]) == [6, 3, 1])
        #expect(Tensor<Double>.defaultStrides(for: [5, 3, 1, 4, 9]) == [108, 36 ,36, 9, 1])
        #expect(Tensor<Double>.defaultStrides(for: [1, 1]) == [1, 1])
    }
}
