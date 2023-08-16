import AlgebraKit

enum Operation {
    case add
    case mul
}


class Value: Hashable, Equatable {
    static func == (lhs: Value, rhs: Value) -> Bool {
        <#code#>
    }

    public var data: Matrix
    public var grad: Matrix
    public let prev: [Value]
    public let op: Operation?
    public var _backward: (() -> Void)?

    public init(data: Matrix, prev: [Value] = [], op: Operation? = nil) {
        self.data = data
        self.op = op
        self.prev = prev
        grad = Matrix(as: data, repeating: 0)
    }

    static func +(lhs: Value, rhs: Value) -> Value {
        let out = Value(data: lhs.data + rhs.data, prev: [lhs, rhs], op: .mul)
        out._backward = {
            lhs.grad = lhs.grad + out.grad
            rhs.grad = rhs.grad + out.grad
        }
        return out
    }

    static func *(lhs: Value, rhs: Value) -> Value {
        let out = Value(data: lhs.data * rhs.data, prev: [lhs, rhs], op: .add)
        out._backward = {
            lhs.grad = lhs.grad + rhs.data * out.grad
            rhs.grad = rhs.grad + lhs.data * out.grad
        }
        return out
    }

    func backward() {
        var topo: [Value] = []
        var visited: Set<Value> = .init()
        func buildTopo(_ v: Value) {
            guard !visited.contains(v) else { return }
            visited.insert(v)
            for child in v.prev {
                buildTopo(child)
            }
            topo.append(v)
        }
        buildTopo(self)
        self.grad = Matrix(as: data, repeating: 1.0)
        for v in topo.reversed() {
            v._backward?()
        }
    }


//    // FYI: eta is just typable version of η.
//    public func applyGrad(eta: Float) {
//        data = data + grad * eta
//    }
}
