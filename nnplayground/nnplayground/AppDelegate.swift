import Cocoa

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var windowController: NSWindowController!

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        let window = NSWindow()
        window.styleMask = [.closable, .miniaturizable, .titled]
        self.window = window

        let windowController = NSWindowController(window: window)
        self.windowController = windowController
        windowController.window?.makeKeyAndOrderFront(nil)
        window.center()
        DispatchQueue.global().async {
            self.doWork()
        }
    }

    private func doWork() {
        guard let modelPath = Bundle.main.object(forInfoDictionaryKey: "MNISTModelPath") as? String else {
            fatalError("MNISTModelPath is not set in Info.plist")
        }
        do {
            let dataset = try MNISTLoader.load(from: URL(fileURLWithPath: modelPath))
            let model = MNISTMLP()
            try model.train(with: dataset)
        } catch {
            print(error)
        }
    }
}
