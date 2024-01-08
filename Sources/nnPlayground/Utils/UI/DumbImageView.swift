import AppKit

final class DumbImageView: NSView {
    var image: NSImage?

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)

        image?.draw(in: dirtyRect)
    }
}
