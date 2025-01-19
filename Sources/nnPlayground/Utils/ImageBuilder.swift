import AppKit
import CoreGraphics
import UniformTypeIdentifiers

// TODO: Update for tensors.
//final class ImageBuilder {
//    // Values in [0, 1] are expected in the matricies.
//    static func buildImage(from matricies: [Matrix], colorTransform: ((UInt8, UInt8, UInt8) -> UInt32)? = nil) -> NSImage? {
//        guard matricies.count == 1 || matricies.count == 3 else {
//            fatalError("One or 3 input matricies are supported.")
//        }
//        var size: Size!
//        for matrix in matricies {
//            guard let size else {
//                size = matrix.size
//                continue
//            }
//            guard size == matrix.size else {
//                fatalError("One of the matricies has wrong size.")
//            }
//        }
//        let isColored = matricies.count == 3
//
//        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
//        let width = size.cols
//        let height = size.rows
//        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
//        let context = CGContext(
//            data: nil,
//            width: width,
//            height: height,
//            bitsPerComponent: 8,
//            bytesPerRow: width * 4,
//            space: colorSpace,
//            bitmapInfo: bitmapInfo.rawValue
//        )
//
//        guard let data = context?.data else { return nil }
//
//        for row in 0..<size.rows {
//            for col in 0..<size.cols {
//                let b = UInt8(matricies[isColored ? 0 : 0][row, col] * 255.0)
//                let g = UInt8(matricies[isColored ? 1 : 0][row, col] * 255.0)
//                let r = UInt8(matricies[isColored ? 2 : 0][row, col] * 255.0)
//
//                let color = colorTransform?(r, g, b) ?? (isColored ? colored(r, g, b) : grayscale(r) )
//                withUnsafePointer(to: color) { ptr in
//                    (data + row * size.cols * 4 + col * 4).copyMemory(from: ptr, byteCount: 4)
//                }
//            }
//        }
//        let cgImage = context?.makeImage()
//        let result = cgImage.map {
//            let nsImage = NSImage(size: NSSize(width: width, height: height))
//            nsImage.addRepresentation(NSBitmapImageRep(cgImage: $0))
//            return nsImage
//        }
//        return result
//    }
//
//    static func grayscale(_ byte: UInt8) -> UInt32 {
//        UInt32(littleEndianBytes: [byte, byte, byte, 0xFF])
//    }
//
//    static func colored(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> UInt32 {
//        UInt32(littleEndianBytes: [b, g, r, 0xFF])
//    }
//}
