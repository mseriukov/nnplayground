import Foundation

extension Array where Element: BinaryFloatingPoint {
    mutating func normalize() {
        let mean = reduce(Element(0), +) / Element(count)
        let diffsq = map({ ($0 - mean) * ($0 - mean) })
        let std_ = diffsq.reduce(Element(0), +) / Element(count)
        let std = sqrt(std_)
        self = map { ($0 - mean) / std }
    }

    mutating func invert() {
        self = map { -$0 }
    }

    init(
        count: Int,
        mean: Float = 0.0,
        std: Float = 0.0,
        seed: UInt32 = 1
    ) {
        var result: [Element] = Array(repeating: 0, count: count)
        let rng = NormalRandomGenerator(mean: 0, std: 1, seed: seed)

        for i in 0..<count {
            result[i] = Element(rng.next())
        }
        self = result
    }
}

extension MutableCollection {
    mutating func mapInPlace(_ x: (inout Element) -> ()) {
        for i in indices {
            x(&self[i])
        }
    }
}
