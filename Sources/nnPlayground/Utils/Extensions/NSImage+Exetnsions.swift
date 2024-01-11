import Foundation
import AppKit

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
}
