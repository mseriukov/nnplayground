extension Tensor {
    public static func identity(rank: Int, size: Int) -> Self {
        var result = Self(zeros: Array(repeating: size, count: rank))
        for i in 0..<size {
            result.assign(1, at: Array(repeating: i, count: rank))
        }
        return result
    }
}
