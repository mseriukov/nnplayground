import ArgumentParser
import Foundation
import AppKit
import Tensor

@main
struct nnplayground: ParsableCommand {
    @Argument(
        help: "Input file path.",
        completion: .file(),
        transform: URL.init(fileURLWithPath:)
    )
    var modelURL: URL? = nil

    mutating func run() throws {
        guard let modelURL else { return }

        getArgumentAndRun(modelURL: modelURL)
    }

    func getArgumentAndRun(modelURL: URL) {
        do {
            let dataset = try MNISTLoader.load(from: modelURL)
            let model = MNISTMLP()
            try model.train(with: dataset)
        } catch {
            print(error)
        }
    }
}
