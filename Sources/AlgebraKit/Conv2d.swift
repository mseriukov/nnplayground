public struct FilterDescriptor {
    public enum Padding {
        /// No padding.
        case valid
        /// Padding to get output of the same size as input.
        case same
        /// Padding to let filter see every piece of input.
        case full
    }
    public let rows: Int
    public let cols: Int
    public let padding: Padding
    public let stride: Int

    public init(rows: Int, cols: Int, padding: Padding, stride: Int) {
        self.rows = rows
        self.cols = cols
        self.padding = padding
        self.stride = stride
    }
}

public func conv2d(
    _ input: Matrix,
    _ filter: Matrix,
    _ filterDescriptor: FilterDescriptor
) -> Matrix {


    return Matrix(rows: 1, cols: 1)
}
