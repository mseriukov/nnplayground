import Foundation
import cnnutils

public class NormalRandomGenerator {
    private var state: random_state_t
    let mean: Float
    let std: Float
    private let lock = UnfairLock()

    public init(mean: Float, std: Float, seed: UInt32) {
        self.state = random_state_t(prev: seed)
        self.mean = mean
        self.std = std
    }

    public func next() -> Float {
        lock.locked {
            random_normal(&state, mean, std)
        }
    }
}
