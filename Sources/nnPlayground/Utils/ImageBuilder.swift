import AppKit
import AlgebraKit
import CoreGraphics
import UniformTypeIdentifiers


final class ImageBuilder {

    static func buildImage(from matrix: Matrix, colorTransform: ((UInt8) -> UInt32)? = nil) -> NSImage? {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let width = matrix.size.cols
        let height = matrix.size.rows
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        )

        guard let data = context?.data else { return nil }

        let mind = matrix.storage.min()!
        let maxd = matrix.storage.max()!
        let offset = mind
        let length = maxd - mind

        for row in 0..<matrix.size.rows {
            for col in 0..<matrix.size.cols {
                let val = UInt8((matrix[row, col] - offset) / length * 255.0)
                let color = colorTransform?(val) ?? grayscale(val)
                withUnsafePointer(to: color) { ptr in
                    (data + row * matrix.size.cols * 4 + col * 4).copyMemory(from: ptr, byteCount: 4)
                }
            }
        }
        let cgImage = context?.makeImage()
        let result = cgImage.map {
            let nsImage = NSImage(size: NSSize(width: width, height: height))
            nsImage.addRepresentation(NSBitmapImageRep(cgImage: $0))
            return nsImage
        }
        return result
    }

    static func grayscale(_ byte: UInt8) -> UInt32 {
        UInt32(littleEndianBytes: [0xFF, byte, byte, byte])
    }
}
