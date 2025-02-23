public final class TensorStorage<Element> {
    public var buffer: UnsafeMutableBufferPointer<Element>

    public init(buffer: UnsafeMutableBufferPointer<Element>) {
        self.buffer = buffer
    }

    public init(repeating value: Element, count: Int) {
        self.buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        for i in 0..<count {
            buffer[i] = value
        }
    }

    deinit {
        buffer.deallocate()
    }

    public subscript(index: Int) -> Element {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }

    public init(_ data: [Element]) {
        self.buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: data.count)
        _ = self.buffer.update(from: data)     
    }

    public func copy() -> TensorStorage {
        let newBuffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: buffer.count)
        _ = newBuffer.update(from: buffer)
        return TensorStorage(buffer: newBuffer)
    }
}

extension TensorStorage: ExpressibleByArrayLiteral {
    public convenience init(arrayLiteral elements: Element...) {
        self.init(elements)
    }
}
