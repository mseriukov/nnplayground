import ArgumentParser
import Foundation

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
        do {
            try MLPTests().run(url: url, testURL: testURL)
        } catch {
            print(error)
        }
    }
}

