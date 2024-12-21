class ArrayStorage<Element>: LinearStorageType {
    private var storage: [Element]

    var size: Int { storage.count }

    init(_ storage: [Element]) {
        self.storage = storage
    }

    subscript(_ index: Int) -> Element {
        get {
            storage[index]
        }
        set {
            storage[index] = newValue
        }
    }
}
