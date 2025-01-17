import ArgumentParser
import Foundation
import AppKit
import AlgebraKit

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

        getArgumentAndRun(inputURL: inputURL, testURL: testURL)
    }

    func getArgumentAndRun(inputURL: URL, testURL: URL) {
        do {
            //try MLPTests().run(url: inputURL, testURL: testURL)
            try TensorMLPTests().train(inputURL: inputURL)
        } catch {
            print("Failed with error: \(error)")
        }
    }
}
