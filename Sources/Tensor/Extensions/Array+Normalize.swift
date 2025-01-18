import Foundation

extension Array where Element: BinaryFloatingPoint {
    public mutating func scaleToUnitInterval() {
        guard let minVal = self.min(), let maxVal = self.max(), minVal != maxVal else {
            self = Array(repeating: 0.5, count: count)
            return
        }
        self = map { ($0 - minVal) / (maxVal - minVal) }
    }

    public mutating func normalize() {
        let mean = reduce(Element(0), +) / Element(count)
        let diffsq = map({ ($0 - mean) * ($0 - mean) })
        let std_ = diffsq.reduce(Element(0), +) / Element(count)
        let std = sqrt(std_)
        self = map { ($0 - mean) / std }
    }

    mutating func invert() {
        self = map { -$0 }
    }
}
