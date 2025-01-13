struct TensorIndexSequence: Sequence, IteratorProtocol {
    let shape: [Int]
    private var indicies: [Int] = []

    init(shape: [Int]) {
        self.shape = shape
    }

    mutating func next() -> [Int]? {
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
