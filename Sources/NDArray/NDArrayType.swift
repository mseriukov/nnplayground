public protocol NDArrayType {
    associatedtype Element
    associatedtype Storage: LinearStorageType where Storage.Element == Element

    var storage: Storage { get }
    var shape: Shape { get }
    subscript(_ s: [Int]) -> Element { get set }
}

extension NDArrayType {
    public subscript(_ s: Int...) -> Element {
        get { return self[s] }
        set(newValue) { self[s] = newValue }
    }
}
