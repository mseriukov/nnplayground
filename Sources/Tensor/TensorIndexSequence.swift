public struct TensorIndexSequence: Sequence, IteratorProtocol {
    let shape: [Int]
    private var indicies: [Int] = []

    public init(shape: [Int]) {
        self.shape = shape
    }

    public mutating func next() -> [Int]? {
        guard !shape.isEmpty else {
            return nil
        }
        
        guard !indicies.isEmpty else {
            indicies = Array(repeating: 0, count: shape.count)
            return indicies
        }
        var index = indicies.count - 1
        var result: [Int]?
        while true {
            if indicies[index] + 1 < shape[index] {
                indicies[index] += 1
                result = indicies
                break
            }
            if index == 0 {
                break
            }
            indicies[index] = 0
            index -= 1
        }
        return result
    }
}
