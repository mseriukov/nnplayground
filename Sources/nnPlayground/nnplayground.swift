import ArgumentParser
import Foundation
import AppKit
import SwiftUI

var _appDelegate: AppDelegate?

@main
struct nnplayground: ParsableCommand {
    @Argument(
        help: "Input file path.",
        completion: .file(),
        transform: URL.init(fileURLWithPath:)
    )
    var inputFile: URL? = nil

    @Argument(
        help: "Input file path.",
        completion: .file(),
        transform: URL.init(fileURLWithPath:)
    )
    var testFile: URL? = nil

    mutating func run() throws {
        guard let url = inputFile, let testURL = testFile else { return }
        let app = NSApplication.shared
        let appDelegate = AppDelegate(url: url, testURL: testURL)
        app.delegate = appDelegate
        _appDelegate = appDelegate
        app.run()
    }
}
