import AlgebraKit

public class Filter {
    public var weight: Parameter
    public var bias: Parameter

    init(rows: Int, cols: Int) {
        weight = Parameter(size: Size(rows, cols))
        bias = Parameter(size: 1)
    }
}

public class Conv2DLayer: Layer {
    public var input: Matrix = .zero
    public var output: Matrix = .zero
    public var parameters: [Parameter] = []
    public var filters: [Filter] = []
    public let filterDescriptor: FilterDescriptor

    init(
        input: Matrix,
        filterDescriptor: FilterDescriptor,
        filterCount: Int
    ) {
        self.filterDescriptor = filterDescriptor       
    }

    public func forward(_ input: Matrix) -> Matrix {
        .zero
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        .zero
    }
}
