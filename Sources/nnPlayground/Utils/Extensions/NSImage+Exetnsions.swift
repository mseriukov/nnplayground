import Foundation
import AppKit
import AlgebraKit

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

    func asMatrix() -> [Matrix] {
        guard let rep = self.representations[0] as? NSBitmapImageRep else {
            fatalError("Image is not a bitmap.")
        }
        let width = rep.pixelsWide
        let height = rep.pixelsHigh

        let p = UnsafeMutableBufferPointer(start: rep.bitmapData, count: width * height)
        let arr = Array(p)

        return [Matrix(size: Size(height, width), data: arr.map { Float($0) / 255.0 })]
    }
}
