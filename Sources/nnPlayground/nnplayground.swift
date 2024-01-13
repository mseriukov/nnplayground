import ArgumentParser
import Foundation
import AppKit

@main
struct nnplayground: ParsableCommand {
    @Argument(
        help: "Input file path.",
        completion: .file(),
        transform: URL.init(fileURLWithPath:)
    )
    var inputURL: URL? = nil

    @Argument(
        help: "Input file path.",
        completion: .file(),
        transform: URL.init(fileURLWithPath:)
    )
    var testURL: URL? = nil

    mutating func run() throws {
        guard let inputURL, let testURL else { return }
        Monitor.shared.run { monitor in
            let display1 = Monitor.shared.addDisplay(title: "Test1", size: .zero)
            let display2 = Monitor.shared.addDisplay(title: "Test2", size: .zero)
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try MLPTests().run(url: inputURL, testURL: testURL) { image in
                        DispatchQueue.main.async {
                            guard let image else { return }
                            display1.setImage(image)
                            display2.setImage(image)
                        }
                    }
                } catch {}
            }
        }
    }
}
