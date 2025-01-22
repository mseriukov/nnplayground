import AppKit

struct TestImage {
    static var cats_gs: NSImage {
        loadEmbeddedImage("cats_gs")
    }

    static var cats: NSImage {
        loadEmbeddedImage("cats")
    }

    private static func loadEmbeddedImage(_ name: String) -> NSImage {
        guard let image = NSImage(data: MachOSectionReader.getEmbeddedData(name)) else {
            fatalError("Cant load embedded image")
        }
        return image
    }
}
