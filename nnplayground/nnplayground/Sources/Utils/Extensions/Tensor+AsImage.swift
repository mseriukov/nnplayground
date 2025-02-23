import AppKit
import CoreGraphics
import UniformTypeIdentifiers
import Tensor

extension Tensor {
    /// Creates `NSImage` from a tensor containing values in 0...255 range.
    /// - Parameter colorTransform: Optional closure converting r, g, b channels into a `UInt32` ARGB value with opaque alpha.
    /// - Returns: An image.
    func asImage(colorTransform: ((UInt8, UInt8, UInt8) -> UInt32)? = nil) -> NSImage? {
        guard rank == 2 || rank == 3 else {
            fatalError("Input tensor shape is not supported.")
        }
        let tensor = self //.unsqueezed(axis: 0)

        let channels = tensor.shape[0]
        let rows = tensor.shape[1]
        let cols = tensor.shape[2]
        guard channels == 1 || channels == 3 else {
            fatalError("One or 3 input channels are supported.")
        }
        let isColored = channels > 1

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let width = cols
        let height = rows
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

        func grayscale(_ byte: UInt8) -> UInt32 {
            UInt32(littleEndianBytes: [byte, byte, byte, 0xFF])
        }

        func colored(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> UInt32 {
            UInt32(littleEndianBytes: [b, g, r, 0xFF])
        }

        for row in 0..<rows {
            for col in 0..<cols {
                let b = UInt8(tensor[isColored ? 0 : 0, row, col])
                let g = UInt8(tensor[isColored ? 1 : 0, row, col])
                let r = UInt8(tensor[isColored ? 2 : 0, row, col])

                let color = colorTransform?(r, g, b) ?? (isColored ? colored(r, g, b) : grayscale(r) )
                withUnsafePointer(to: color) { ptr in
                    (data + row * cols * 4 + col * 4).copyMemory(from: ptr, byteCount: 4)
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
