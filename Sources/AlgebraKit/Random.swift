import Foundation
import cnnutils

public enum Random {
    static func setSeed(_ seed: UInt32) {
        random_set_seed(seed)
    }

    static func uniform() -> Float {
        random_uniform()
    }

    static func normal(mean: Float, stdDev: Float) -> Float {
        random_normal(mean, stdDev)
    }
}
