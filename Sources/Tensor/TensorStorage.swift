public enum Diagnostics {
    public nonisolated(unsafe) static var totalSize: UInt64 = 0
}

public final class TensorStorage<Element> {
    public var buffer: UnsafeMutableBufferPointer<Element>

    public init(buffer: UnsafeMutableBufferPointer<Element>) {
        self.buffer = buffer
        Diagnostics.totalSize += UInt64(buffer.count)
    }

    public init(repeating value: Element, count: Int) {
        self.buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        for i in 0..<count {
            buffer[i] = value
        }
        Diagnostics.totalSize += UInt64(buffer.count)
    }

    deinit {
        buffer.deallocate()
        Diagnostics.totalSize -= UInt64(buffer.count)
    }

    public subscript(index: Int) -> Element {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }

    public init(_ data: [Element]) {
        self.buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: data.count)
        _ = self.buffer.update(from: data)
        Diagnostics.totalSize += UInt64(buffer.count)
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
