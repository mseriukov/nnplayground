import XCTest
import AlgebraKit
@testable import NNKit

final class BackpropTests: XCTestCase {

    var nn: [Layer] = []
    
    override func setUp(){
       super.setUp()


    }

    func test_forward() throws {
        var input = Matrix(rows: 1, cols: 2, data: [
            2.0, 3.0
        ])
        for l in nn {
            input = l.forward(input)
        }
        XCTAssertEqual(input.storage[0], 0.18, accuracy: 0.001)
    }

    func test_backward() throws {
//        var input = Matrix(rows: 1, cols: 2, data: [
//            2.0, 3.0
//        ])
//        for l in nn {
//            l.forward(input: input)
//            input = l.output
//        }
//        var delta = Matrix(rows: 1, cols: 1, repeating: 0.191 - 1)
//        for l in nn.reversed() {
//            delta = l.backward(localGradient: delta)
//        }
//
//        print(nn[0].wgrad)
//        print(nn[1].wgrad)

        //XCTAssertEqual(input.storage, [0.2533])
    }
}
