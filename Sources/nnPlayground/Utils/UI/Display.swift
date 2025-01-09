import AppKit
import SnapKit

@MainActor
final class Display: NSObject, NSWindowDelegate {
    private let window: NSWindow
    private weak var imageView: NSImageView?

    var onClose: (() -> Void)?

    var title: String = "" {
        didSet {
            update()
        }
    }

    var size: CGSize = .zero {
        didSet {
            update()
        }
    }

    override init() {
        window = NSWindow()
        window.styleMask = [.closable, .miniaturizable, .titled]

        let imageView = NSImageView()

        let contentView = NSView()
        window.contentView = contentView
        contentView.addSubview(imageView)
        imageView.snp.makeConstraints { $0.edges.equalToSuperview() }
        self.imageView = imageView

        super.init()

        window.delegate = self
        window.center()
        window.makeKeyAndOrderFront(window)
    }

    func windowWillClose(_ notification: Notification) {
        onClose?()
    }

    public func setImage(_ image: NSImage?) {
        guard let image else {
            imageView?.image = nil
            return
        }
        let rep = image.representations[0]
        let imageSize = NSSize(width: rep.pixelsWide, height: rep.pixelsHigh)
        // Yep. Divine API...
        image.size = imageSize
        window.setContentSize(imageSize)
        imageView?.image = image
    }

    private func update() {
        window.title = title
        window.setContentSize(size)
    }
}
