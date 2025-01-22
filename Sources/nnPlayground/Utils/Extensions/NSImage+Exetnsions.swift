import Foundation
import AppKit
import Tensor

extension NSImage {
    enum NSImageSavingError: Error {
        case failedToCreateCGImage
        case failedToGetPNGData
    }

    func save(to url: URL) throws {
        guard let cgImage = cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSImageSavingError.failedToCreateCGImage
        }
        let newRep = NSBitmapImageRep(cgImage: cgImage)
        newRep.size = size
        guard let pngData = newRep.representation(using: .png, properties: [:]) else {
            throw NSImageSavingError.failedToGetPNGData
        }
        try pngData.write(to: url)
    }

    func asTensor() -> Tensor {
        guard let rep = self.representations[0] as? NSBitmapImageRep else {
            fatalError("Image is not a bitmap.")
        }
        let width = rep.pixelsWide
        let height = rep.pixelsHigh
        let bpp = rep.bitsPerPixel

        let arr = Array(UnsafeMutableBufferPointer(start: rep.bitmapData, count: width * height * bpp / 8))
        if bpp == 8 {
            return Tensor([1, height, width], arr.map { Tensor.Element($0) })
        } else if bpp == 32 {
            var rdata = Array<Float>()
            var gdata = Array<Float>()
            var bdata = Array<Float>()
            for (i, e) in arr.enumerated() {
                let val = Float(e)
                switch i % 4 {
                case 0: rdata.append(val)
                case 1: gdata.append(val)
                case 2: bdata.append(val)
                default: break
                }
            }
            return Tensor([3, height, width], rdata + gdata + bdata)
        }
        fatalError("Unsupported image format")
    }
}
