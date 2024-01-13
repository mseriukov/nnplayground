import AppKit
import SnapKit

final class Display: NSObject, NSWindowDelegate {
    private let window: NSWindow
    private weak var imageView: DumbImageView?

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

        let imageView = DumbImageView()
        window.contentView!.addSubview(imageView)
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

    public func setImage(_ image: NSImage) {
        window.setContentSize(image.size)
        imageView?.image = image
    }

    private func update() {
        window.title = title
        window.setContentSize(size)
    }
}
