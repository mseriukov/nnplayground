extension Matrix {
    public static let laplacian5x5: Matrix = {
        Matrix(rows: 5, cols: 5, data: [
             0.0,  0.0, -1.0,  0.0,  0.0,
             0.0, -1.0, -2.0, -1.0,  0.0,
            -1.0, -2.0, 16.0, -2.0, -1.0,
             0.0, -1.0, -2.0, -1.0,  0.0,
             0.0,  0.0, -1.0,  0.0,  0.0,
        ])
    }()

    public static let laplacian3x3: Matrix = {
        Matrix(rows: 3, cols: 3, data: [
             0.0,  1.0, 0.0,
             1.0, -4.0, 1.0,
             0.0,  1.0, 0.0,
        ])
    }()

    public static let gaussian3x3: Matrix = {
        Matrix(rows: 3, cols: 3, data: [
             1.0,  2.0, 1.0,
             2.0,  4.0, 2.0,
             1.0,  2.0, 1.0,
        ]) / 16.0
    }()

    public static let gaussian5x5: Matrix = {
        Matrix(rows: 5, cols: 5, data: [
             1.0,  4.0,  7.0,  4.0, 1.0,
             4.0, 16.0, 26.0, 16.0, 4.0,
             7.0, 26.0, 41.0, 26.0, 7.0,
             4.0, 16.0, 26.0, 16.0, 4.0,
             1.0,  4.0,  7.0,  4.0, 1.0,
        ]) / 273.0
    }()

    public static let average3x3: Matrix = {
        Matrix(rows: 3, cols: 3, data: [
             1.0,  1.0, 1.0,
             1.0,  1.0, 1.0,
             1.0,  1.0, 1.0,
        ]) / 9.0
    }()
}
