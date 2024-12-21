import AppKit
import AlgebraKit
import CoreGraphics
import UniformTypeIdentifiers


final class ImageBuilder {

    static func buildImage(from matrix: Matrix) -> NSImage? {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let width = matrix.cols
        let height = matrix.rows
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
        let offset = abs(mind)
        let length = abs(mind) + abs(maxd)

        for row in 0..<matrix.rows {
            for col in 0..<matrix.cols {
                let val = UInt8((matrix[row, col] + offset) / length * 255.0)
                let color = Viridis.color(val)
                withUnsafePointer(to: color) { ptr in
                    (data + row * matrix.cols * 4 + col * 4).copyMemory(from: ptr, byteCount: 4)
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
}
