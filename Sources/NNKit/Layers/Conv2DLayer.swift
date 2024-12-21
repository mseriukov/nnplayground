import AlgebraKit

public class Filter {
    public var weight: Parameter
    public var bias: Parameter

    init(rows: Int, cols: Int) {
        weight = Parameter(rows: rows, cols: cols)
        bias = Parameter(rows: 1, cols: 1)
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
        for _ in 0..<filterCount {
            let filter = Filter(
                rows: filterDescriptor.rows,
                cols: filterDescriptor.cols
            )
            filters.append(filter)
            parameters.append(filter.weight)
            parameters.append(filter.bias)
        }
    }

    public func forward(_ input: Matrix) -> Matrix {
        .zero
    }

    public func backward(_ localGradient: Matrix) -> Matrix {
        .zero
    }
}
