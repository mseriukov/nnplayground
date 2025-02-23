import AppKit

let mainMenu: NSMenu = {
    let appMenu = NSMenuItem()
    appMenu.submenu = NSMenu()
    appMenu.submenu?.addItem(
        NSMenuItem(
            title: "Quit",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
    )
    let mainMenu = NSMenu(title: "")
    mainMenu.addItem(appMenu)
    return mainMenu
} ()

let app = NSApplication.shared
let appDelegate = AppDelegate()

app.delegate = appDelegate
app.mainMenu = mainMenu
app.setActivationPolicy(.regular)
app.activate(ignoringOtherApps: true)
app.run()
