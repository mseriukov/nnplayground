import AppKit
import SnapKit

class Monitor: NSObject, NSApplicationDelegate {
    public static var shared = Monitor()
    public private(set) var displays: [Display] = []

    private var onFinishLaunching: ((Monitor) -> Void)?
    private var mainMenu: NSMenu?

    public func run(_ onFinishLaunching: ((Monitor) -> Void)?) {
        self.onFinishLaunching = onFinishLaunching
        let app = NSApplication.shared
        app.delegate = Self.shared
        app.run()
    }

    public func addDisplay(title: String, size: CGSize) -> Display {
        let display = Display()
        display.title = title
        display.size = size
        display.onClose = {
            self.displays = self.displays.filter { $0 !== display }
            if self.displays.isEmpty {
                NSApplication.shared.terminate(0)
            }
        }
        displays.append(display)
        return display
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let mainMenu = buildMainMenu()
        NSApplication.shared.mainMenu = mainMenu
        self.mainMenu = mainMenu

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
