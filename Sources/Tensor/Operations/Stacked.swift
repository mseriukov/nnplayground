extension Tensor {
    public static func stacked(_ arr: [Tensor]) -> Tensor {
        let shape = arr[0].shape
        let batchShape = [arr.count] + shape
        return Tensor(batchShape, arr.flatMap { $0.storage.data })
    }
}
