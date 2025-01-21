import Foundation
import Tensor

public enum SafeTensorDType: String, Codable {
    case float32 = "F32"
}

/// SafeTensors metadata structure
public struct SafeTensorsMetadata: Codable {
    struct TensorInfo: Codable {
        let dtype: SafeTensorDType
        let shape: [Int]
        let data_offsets: [Int] // Start and end byte offsets in the binary data
    }
    var tensors: [String: TensorInfo]
}

public enum SafeTensorsError: Error {
    case failedToEncodeMetadata
    case failedToOpenFile
}

public struct SafeTensors {
    public static func loadMemoryMapped(from url: URL) throws -> [String: Tensor] {
        let fileHandle = try FileHandle(forReadingFrom: url)
        let fileSize = fileHandle.seekToEndOfFile()

        let mmapData = try Data(contentsOf: url, options: .mappedIfSafe)

        let headerLenData = mmapData.prefix(8)
        let headerLen = headerLenData.withUnsafeBytes { $0.load(as: Int64.self) }.littleEndian

        let binaryDataStart = 8 + headerLen
        let jsonData = mmapData[8..<(8 + headerLen)]

        let metadata = try JSONDecoder().decode(SafeTensorsMetadata.self, from: jsonData)

        var tensors: [String: Tensor] = [:]
        for (name, info) in metadata.tensors {
            let start = info.data_offsets[0] + Int(binaryDataStart)
            let end = info.data_offsets[1] + Int(binaryDataStart)

            guard start < end, end <= mmapData.count else {
                print("Invalid tensor data offsets for \(name)")
                continue
            }

            let tensorBytes = mmapData[start..<end]
            let values: [Tensor.Element] = tensorBytes.withUnsafeBytes { Array($0.bindMemory(to: Tensor.Element.self)) }
            tensors[name] = Tensor(info.shape, values)
        }
        return tensors
    }

    public static func save(to url: URL, tensors: [String: Tensor]) throws {
        var metadata = SafeTensorsMetadata(tensors: [:])
        var binaryData = Data()

        for (name, tensor) in tensors {
            let startOffset = binaryData.count
            binaryData.append(tensor.storage.data.withUnsafeBufferPointer { Data(buffer: $0) })
            let endOffset = binaryData.count

            metadata.tensors[name] = SafeTensorsMetadata.TensorInfo(
                dtype: .float32, shape: tensor.shape, data_offsets: [startOffset, endOffset]
            )
        }

        guard let jsonData = try? JSONEncoder().encode(metadata) else {
            throw SafeTensorsError.failedToEncodeMetadata
        }

        let jsonDataLen: Int64 = Int64(jsonData.count)

        var resultData = Data()

        withUnsafeBytes(of: jsonDataLen) { resultData.append(contentsOf: $0) }
        resultData.append(jsonData)
        resultData.append(binaryData)

        try resultData.write(to: url)
    }
}
