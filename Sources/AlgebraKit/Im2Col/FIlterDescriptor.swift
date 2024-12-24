public struct FilterDescriptor: Hashable {
    public var size: Size
    public var dilation: Dilation

    public var effectiveSize: Size {
        Size(
            dilation.vertical * (size.rows - 1) + 1,
            dilation.horizontal * (size.cols - 1) + 1
        )
    }

    public init(
        size: Size,
        dilation: Dilation = .none
    ) {
        self.size = size
        self.dilation = dilation
    }
}
