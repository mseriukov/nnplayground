import AppKit
import SnapKit

@available(macOS 10.15, *)
class WindowDelegate: NSObject, NSWindowDelegate {

    func windowWillClose(_ notification: Notification) {
        NSApplication.shared.terminate(0)
    }
}

@available(macOS 10.15, *)
class Monitor: NSObject, NSApplicationDelegate {
    public static var shared = Monitor()

    private let window = NSWindow()
    private let windowDelegate = WindowDelegate()
    private weak var imageView: DumbImageView?
    private var onFinishLaunching: ((Monitor) -> Void)?

    public func run(_ onFinishLaunching: ((Monitor) -> Void)?) {
        self.onFinishLaunching = onFinishLaunching
        let app = NSApplication.shared
        app.delegate = Self.shared
        app.run()
    }

    public func setContentSize( _ size: CGSize) {
        window.setContentSize(size)
    }

    public func setImage(_ image: NSImage) {
        window.setContentSize(image.size)
        imageView?.image = image
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let appMenu = NSMenuItem()
        appMenu.submenu = NSMenu()
        appMenu.submenu?.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        let mainMenu = NSMenu(title: "My Swift Script")
        mainMenu.addItem(appMenu)
        NSApplication.shared.mainMenu = mainMenu

        setContentSize(CGSize(width: 480, height: 270))
        window.styleMask = [.closable, .miniaturizable, .titled]
        window.delegate = windowDelegate
        window.title = "NNPlayground"

        let imageView = DumbImageView()
        window.contentView!.addSubview(imageView)
        imageView.snp.makeConstraints { $0.edges.equalToSuperview() }
        self.imageView = imageView

        window.center()
        window.makeKeyAndOrderFront(window)

        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)

        onFinishLaunching?(self)
    }
}
