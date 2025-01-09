struct DefaultStridesGenerator {
    static func defaultStrides(for shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        for i in (0..<(shape.count - 1)).reversed() {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        return strides
    }
}
