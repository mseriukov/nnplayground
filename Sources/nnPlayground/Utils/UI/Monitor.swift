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
    private var mainMenu: NSMenu?

    var title: String = "" {
        didSet {
            updateTitle()
        }
    }

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

    private func updateTitle() {
        mainMenu?.title = title
        window.title = title
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let mainMenu = buildMainMenu()
        NSApplication.shared.mainMenu = mainMenu
        self.mainMenu = mainMenu

        setContentSize(CGSize(width: 200, height: 200))

        window.styleMask = [.closable, .miniaturizable, .titled]
        window.delegate = windowDelegate

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

    private func buildMainMenu() -> NSMenu {
        let appMenu = NSMenuItem()
        appMenu.submenu = NSMenu()
        appMenu.submenu?.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        let mainMenu = NSMenu(title: "")
        mainMenu.addItem(appMenu)
        return mainMenu
    }
}
