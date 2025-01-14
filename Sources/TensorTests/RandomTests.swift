import Testing
@testable import Tensor

@Suite
struct RandomTests {
    @Test
    // Just to peek at results
    func test() throws {
        var generator: any RandomNumberGenerator = SeedableRandomNumberGenerator(seed: 42)
        let a = Tensor.random(shape: [3, 3], distribution: .uniform(lowerBound: 0, upperBound: 1), generator: &generator)
        print(a.storage.data)
    }
}
