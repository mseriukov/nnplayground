import Foundation
import Tensor

public struct IDXLoader {
    public enum IDXLoaderError: Error {
        case wrongIDXMAgic
    }

    enum DataType: UInt8 {
        case uint8 = 0x08
        case int8 = 0x09
        case int16 = 0x0B
        case int32 = 0x0C
        case float32 = 0x0D
        case float64 = 0x0E
    }

    public static func load(from url: URL) throws -> Tensor {
        let mmapData = try Data(contentsOf: url, options: .mappedIfSafe)

        let magic = mmapData[0..<2].withUnsafeBytes { $0.load(as: UInt16.self) }.littleEndian
        guard
            magic == 0,
            let type = DataType(rawValue: mmapData[2..<3].withUnsafeBytes { $0.load(as: UInt8.self) })
        else {
            throw IDXLoaderError.wrongIDXMAgic
        }
        let ndims = Int(mmapData[3..<4].withUnsafeBytes { $0.load(as: UInt8.self) })
        let shapedata = mmapData[4..<(4 + ndims * 4)]
        let shape = shapedata.withUnsafeBytes({ Array($0.bindMemory(to: UInt32.self)) }).map { Int($0.bigEndian) }
        let data = mmapData[(4 + ndims * 4)...]

        let values: [Float32] = switch type {
        case .uint8: data.withUnsafeBytes({ Array($0.bindMemory(to: UInt8.self)) }).map { Float32($0) }
        case .int8: data.withUnsafeBytes({ Array($0.bindMemory(to: Int8.self)) }).map { Float32($0) }
        case .int16: data.withUnsafeBytes({ Array($0.bindMemory(to: Int16.self)) }).map { Float32($0.bigEndian) }
        case .int32: data.withUnsafeBytes({ Array($0.bindMemory(to: Int32.self)) }).map { Float32($0.bigEndian) }
        // Most likely wrong endianness.
        case .float32: data.withUnsafeBytes({ Array($0.bindMemory(to: Float32.self)) }).map { $0 }
        case .float64: data.withUnsafeBytes({ Array($0.bindMemory(to: Float64.self)) }).map { Float32($0) }
        }
        return Tensor(shape, values)
    }
}
