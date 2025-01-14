import Testing
@testable import Tensor

@Suite
struct RandomTests {
    @Test
    // Just to peek at results
    func test() throws {
        var generator: any RandomNumberGenerator = SeedableRandomNumberGenerator(seed: 42)
        let a = Tensor<Double>.random(shape: [3, 3], distribution: .kaiming(channels: 32), generator: &generator)
        print(a.storage.data)
    }
}
