import AppKit
import SnapKit

@available(macOS 10.15, *)
class WindowDelegate: NSObject, NSWindowDelegate {

    func windowWillClose(_ notification: Notification) {
        NSApplication.shared.terminate(0)
    }
}

@available(macOS 10.15, *)
class AppDelegate: NSObject, NSApplicationDelegate {
    let window = NSWindow()
    let windowDelegate = WindowDelegate()
    let url: URL
    let testURL: URL

    func setContentSize( _ size: CGSize) {
        window.setContentSize(size)
    }

    private weak var imageView: DumbImageView? {
        didSet {

        }
    }

    init(url: URL, testURL: URL){
        self.url = url
        self.testURL = testURL
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

        DispatchQueue.global().async {
            do {
                try MLPTests().run(url: self.url, testURL: self.testURL) { image in
                    DispatchQueue.main.async {
                        guard let image else { return }
                        self.imageView?.image = image
                        self.setContentSize(image.size)
                    }
                }
            } catch {}
        }
    }
}
