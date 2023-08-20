import AlgebraKit

public class Parameter {
    public var value: Matrix
    public var grad: Matrix

    init(rows: Int, cols: Int) {
        value = Matrix(rows: rows, cols: cols, repeating: 0)
        grad = Matrix(as: value, repeating: 0)
    }
}
