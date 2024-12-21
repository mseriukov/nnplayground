public protocol LinearStorageType: AnyObject {
    associatedtype Element

    var size: Int { get }

    subscript(_ index: Int) -> Element { get set }
}
