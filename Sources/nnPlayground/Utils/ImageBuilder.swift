import AppKit
import AlgebraKit
import CoreGraphics
import UniformTypeIdentifiers


final class ImageBuilder {

    static func buildImage(from matrix: Matrix, saveTo url: URL? = nil) -> NSImage? {
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
                //                AABBGGRR
                var t: UInt32 = 0xff000000
                let val = UInt8((matrix[row, col] + offset) / length * 255.0)
                let color = Viridis.color(val)
                t |= UInt32(max(0, min(color.r, 255))) << 0
                t |= UInt32(max(0, min(color.g, 255))) << 8
                t |= UInt32(max(0, min(color.b, 255))) << 16
                withUnsafePointer(to: t) { ptr in
                    (data + row * matrix.cols * 4 + col * 4).copyMemory(from: ptr, byteCount: 4)
                }
            }
        }
        let cgImage = context?.makeImage()
        if 
            let url,
            let cgImage,
            let dest = CGImageDestinationCreateWithURL(
                url as CFURL,
                UTType.png.identifier as CFString,
                1,
                nil
            )
        {
            CGImageDestinationAddImage(dest, cgImage, nil)
            CGImageDestinationFinalize(dest)
        }
        let result = cgImage.map {
            NSImage(cgImage: $0, size: NSSize(width: width, height: height))
        }
        return result
    }

    static func saveImage(_ image: NSImage, atUrl url: URL) {
        guard
            let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
            else { return } // TODO: handle error
        let newRep = NSBitmapImageRep(cgImage: cgImage)
        newRep.size = image.size // if you want the same size
        guard
            let pngData = newRep.representation(using: .png, properties: [:])
            else { return } // TODO: handle error
        do {
            try pngData.write(to: url)
        }
        catch {
            print("error saving: \(error)")
        }
    }
}
